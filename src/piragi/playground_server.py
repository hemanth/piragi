"""piragi playground server — real execution backend for the web playground."""

import json
import sys
import os
import io
import traceback
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class StreamingWriter(io.TextIOBase):
    """Custom stdout that sends each write() as an SSE event."""

    def __init__(self, wfile):
        self._wfile = wfile
        self._buffer = ""

    def write(self, s):
        if not s:
            return 0
        try:
            payload = json.dumps({"t": s})
            self._wfile.write(f"data: {payload}\n\n".encode())
            self._wfile.flush()
        except Exception:
            pass
        return len(s)

    def flush(self):
        try:
            self._wfile.flush()
        except Exception:
            pass


# Thread-local stdout proxy: library print() calls go to the right SSE stream
_thread_local = threading.local()
_real_stdout = sys.stdout
_real_stderr = sys.stderr


class ThreadLocalStdout(io.TextIOBase):
    """Proxy that delegates to a per-thread writer or falls back to real stdout."""

    def write(self, s):
        writer = getattr(_thread_local, "writer", None)
        if writer:
            return writer.write(s)
        return _real_stdout.write(s)

    def flush(self):
        writer = getattr(_thread_local, "writer", None)
        if writer:
            writer.flush()
        else:
            _real_stdout.flush()

    @property
    def encoding(self):
        return _real_stdout.encoding

    def fileno(self):
        return _real_stdout.fileno()


sys.stdout = ThreadLocalStdout()
sys.stderr = ThreadLocalStdout()


class PlaygroundHandler(SimpleHTTPRequestHandler):
    """Serves playground HTML + handles /api/exec for real Python execution."""

    playground_dir = str(Path(__file__).parent.parent.parent / "playground")
    llm_config = {"model": "llama3.2", "base_url": "", "api_key": "", "ollama": False}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=self.playground_dir, **kwargs)

    def do_POST(self):
        if self.path == "/api/exec":
            self._handle_exec()
        elif self.path == "/api/exec-stream":
            self._handle_exec_stream()
        elif self.path == "/api/health":
            self._json_response({"status": "ok", "engine": "piragi", "llm": self.llm_config})
        elif self.path == "/api/llm-config":
            self._handle_llm_config()
        elif self.path == "/api/save-snippet":
            self._handle_save_snippet()
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path == "/api/health":
            self._json_response({"status": "ok", "engine": "piragi", "llm": self.llm_config})
            return
        if self.path == "/api/llm-config":
            self._json_response(self.llm_config)
            return
        if self.path.startswith("/api/files"):
            self._handle_files()
            return
        super().do_GET()

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json_response(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self._cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_exec(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            code = body.get("code", "")
        except (json.JSONDecodeError, ValueError):
            self._json_response({"error": "Invalid JSON"}, 400)
            return

        # Capture stdout/stderr
        old_stdout, old_stderr = sys.stdout, sys.stderr
        captured = io.StringIO()
        sys.stdout = captured
        sys.stderr = captured

        # Build execution namespace with piragi pre-imported
        namespace = {"__builtins__": __builtins__}
        try:
            import piragi
            from piragi import Ragi, EmbeddingGenerator, AsyncRagi
            from piragi.stores import QdrantStore
            namespace.update({
                "piragi": piragi,
                "Ragi": Ragi,
                "EmbeddingGenerator": EmbeddingGenerator,
                "AsyncRagi": AsyncRagi,
                "QdrantStore": QdrantStore,
            })
        except ImportError:
            pass

        # Inject LLM config so cells can use ask()
        llm = self.llm_config
        if llm.get("base_url") or llm.get("ollama"):
            namespace["_llm_config"] = {
                "model": llm.get("model", "llama3.2"),
                "base_url": llm.get("base_url") or "http://localhost:11434/v1",
                "api_key": llm.get("api_key") or "ollama",
            }

        # Detect shell commands (pip, apt, etc.)
        stripped = code.strip()
        if stripped.startswith(("pip ", "pip3 ", "!")) :
            import subprocess
            cmd = stripped.lstrip("!")
            # Replace bare pip with sys.executable -m pip
            if cmd.startswith(("pip ", "pip3 ")):
                parts = cmd.split(maxsplit=1)
                cmd = f"{sys.executable} -m pip {parts[1] if len(parts) > 1 else ''}"
            try:
                result = subprocess.run(
                    cmd.split(), capture_output=True, text=True, timeout=60
                )
                output = result.stdout + result.stderr
                self._json_response({"output": output or "(done)"})
            except Exception as e:
                self._json_response({"output": "", "error": str(e)})
            return

        error = None
        try:
            exec(compile(code, "<playground>", "exec"), namespace)
        except Exception:
            error = traceback.format_exc()

        sys.stdout, sys.stderr = old_stdout, old_stderr
        output = captured.getvalue()

        if error:
            self._json_response({"output": output, "error": error})
        else:
            self._json_response({"output": output or "(no output)"})

    def _handle_exec_stream(self):
        """Execute code with SSE streaming — each print() sends an event."""
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            code = body.get("code", "")
        except (json.JSONDecodeError, ValueError):
            self._json_response({"error": "Invalid JSON"}, 400)
            return

        # Start SSE response
        self.send_response(200)
        self._cors_headers()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        # Build namespace (same as _handle_exec)
        namespace = {"__builtins__": __builtins__}
        try:
            import piragi
            from piragi import Ragi, EmbeddingGenerator, AsyncRagi
            from piragi.stores import QdrantStore
            namespace.update({
                "piragi": piragi,
                "Ragi": Ragi,
                "EmbeddingGenerator": EmbeddingGenerator,
                "AsyncRagi": AsyncRagi,
                "QdrantStore": QdrantStore,
            })
        except ImportError:
            pass

        llm = self.llm_config
        if llm.get("base_url") or llm.get("ollama"):
            namespace["_llm_config"] = {
                "model": llm.get("model", "llama3.2"),
                "base_url": llm.get("base_url") or "http://localhost:11434/v1",
                "api_key": llm.get("api_key") or "ollama",
            }

        # Set thread-local writer so ALL print() calls stream to this SSE connection
        writer = StreamingWriter(self.wfile)
        _thread_local.writer = writer

        error = None
        try:
            exec(compile(code, "<playground>", "exec"), namespace)
        except Exception:
            error = traceback.format_exc()

        _thread_local.writer = None

        # Send final event
        try:
            if error:
                payload = json.dumps({"error": error})
                self.wfile.write(f"event: error\ndata: {payload}\n\n".encode())
            self.wfile.write(b"event: done\ndata: {}\n\n")
            self.wfile.flush()
        except Exception:
            pass

    def log_message(self, format, *args):
        """Quieter logging."""
        try:
            msg = format % args
            if "/api/" in msg:
                return
        except Exception:
            pass
        super().log_message(format, *args)

    def _handle_files(self):
        import urllib.parse
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        req_path = qs.get("path", [os.getcwd()])[0]
        
        if ".." in req_path:
            self._json_response({"error": "Invalid path"}, 400)
            return
            
        try:
            entries = []
            for item in os.scandir(req_path):
                if item.name.startswith("."):
                    continue
                if item.is_file():
                    ext = os.path.splitext(item.name)[1]
                    entries.append({"name": item.name, "type": "file", "size": item.stat().st_size, "ext": ext})
                elif item.is_dir():
                    try:
                        children = len(os.listdir(item.path))
                    except Exception:
                        children = 0
                    entries.append({"name": item.name, "type": "dir", "children": children})
            self._json_response({"path": req_path, "entries": entries})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _handle_save_snippet(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            filename = body.get("filename", "")
            code = body.get("code", "")
        except Exception:
            self._json_response({"error": "Invalid JSON"}, 400)
            return

        if ".." in filename or "/" in filename or "\\" in filename:
            self._json_response({"error": "Invalid filename"}, 400)
            return
            
        ext = os.path.splitext(filename)[1]
        if ext not in [".py", ".md", ".txt"]:
            self._json_response({"error": "Invalid extension"}, 400)
            return
            
        save_dir = os.path.join(os.getcwd(), "piragi-playground-snippets")
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, filename)
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(code)
            self._json_response({"ok": True, "path": os.path.abspath(save_path)})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _handle_llm_config(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            PlaygroundHandler.llm_config.update(body)
            self._json_response({"ok": True, "config": PlaygroundHandler.llm_config})
        except Exception as e:
            self._json_response({"error": str(e)}, 400)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def _detect_ollama():
    """Check if Ollama is running locally."""
    import urllib.request
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            PlaygroundHandler.llm_config["ollama"] = True
            PlaygroundHandler.llm_config["base_url"] = "http://localhost:11434/v1"
            PlaygroundHandler.llm_config["api_key"] = "ollama"
            if models:
                PlaygroundHandler.llm_config["model"] = models[0]
                PlaygroundHandler.llm_config["available_models"] = models
            print(f"  Ollama detected: {', '.join(models[:5])}")
    except Exception:
        print("  Ollama not detected (ask() will need manual LLM config)")


def start_server(port=8787, open_browser=True):
    """Start the playground server."""
    # Check env vars for LLM config
    if os.environ.get("OPENAI_API_KEY"):
        PlaygroundHandler.llm_config["api_key"] = os.environ["OPENAI_API_KEY"]
    if os.environ.get("OPENAI_BASE_URL"):
        PlaygroundHandler.llm_config["base_url"] = os.environ["OPENAI_BASE_URL"]

    _detect_ollama()

    server = ThreadedHTTPServer(("127.0.0.1", port), PlaygroundHandler)
    url = f"http://localhost:{port}"

    # Preload embedding model in background so first cell run is fast
    def _preload_model():
        try:
            from piragi import EmbeddingGenerator
            EmbeddingGenerator(model="BAAI/bge-small-en-v1.5").embed_query("warmup")
            print("  \033[1;32m✓\033[0m embedding model preloaded")
        except Exception:
            pass

    threading.Thread(target=_preload_model, daemon=True).start()

    print(f"\033[1;36m")
    print(f"  piragi playground running at \033[1;37m{url}\033[1;36m")
    print(f"  Press Ctrl+C to stop\033[0m")
    print()

    if open_browser:
        import webbrowser
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\033[1;33mPlayground stopped.\033[0m")
        server.server_close()
