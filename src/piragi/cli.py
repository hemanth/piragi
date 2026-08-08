"""piragi CLI - interactive REPL for exploring RAG."""
import argparse
import sys
import code

def repl():
    """Start an interactive piragi REPL."""
    try:
        from piragi import Ragi, EmbeddingGenerator, AsyncRagi
    except ImportError as e:
        print(f"Error importing piragi: {e}")
        return

    logo = r"""
     ____  _                  _ 
    / __ \(_)________ _____ _(_)
   / /_/ / / ___/ __ `/ __ `/ / 
  / ____/ / /  / /_/ / /_/ / /  
 /_/   /_/_/   \__,_/\__, /_/   
                    /____/      
"""
    banner = (
        "\033[1;36m" + logo + "\033[0m"
        "Welcome to the \033[1;32mpiragi\033[0m interactive REPL!\n\n"
        "Available objects:\n"
        "  \033[1;33mRagi\033[0m               - Main class for document indexing and retrieval\n"
        "  \033[1;33mEmbeddingGenerator\033[0m - For custom embedding models\n"
        "  \033[1;33mAsyncRagi\033[0m          - Async version of Ragi\n\n"
        "Available functions:\n"
        "  \033[1;34mask(source, question)\033[0m - Quick one-shot question\n"
        "  \033[1;34mindex(source)\033[0m         - Quick document index\n\n"
        "Type \033[1mhelp(Ragi)\033[0m for API details, or press Ctrl+D to exit.\n"
    )

    def ask(source, question, model="all-mpnet-base-v2", backend=None):
        ragi = Ragi(embedding_model=model, embedding_backend=backend)
        ragi.add(source)
        results = ragi.search(question)
        if not results:
            print("\033[1;31mNo relevant documents found.\033[0m")
            return
        
        print(f"\033[1;36mFound {len(results)} relevant chunks.\033[0m")
        print(f"\033[1;35mTop result:\033[0m {results[0].content[:200]}...")

    def index(source, model="all-mpnet-base-v2", backend=None):
        ragi = Ragi(embedding_model=model, embedding_backend=backend)
        ragi.add(source)
        print(f"\033[1;32mSuccessfully indexed {source}\033[0m")
        return ragi

    local_vars = {
        "Ragi": Ragi,
        "EmbeddingGenerator": EmbeddingGenerator,
        "AsyncRagi": AsyncRagi,
        "ask": ask,
        "index": index,
    }

    try:
        import readline
        import rlcompleter
        readline.set_completer(rlcompleter.Completer(local_vars).complete)
        readline.parse_and_bind("tab: complete")
    except ImportError:
        pass

    sys.ps1 = "\033[1;32mpiragi>\033[0m "
    sys.ps2 = "\033[1;32m.......\033[0m "
    
    console = code.InteractiveConsole(local_vars)
    try:
        console.interact(banner, exitmsg="\nGoodbye!")
    except SystemExit:
        pass

def one_shot_ask(args):
    try:
        from piragi import Ragi
    except ImportError as e:
        print(f"Error importing piragi: {e}")
        return

    print(f"\033[1;36mIndexing {args.source}...\033[0m")
    ragi = Ragi(embedding_model=args.model, embedding_backend=args.backend)
    ragi.add(args.source)
    
    print(f"\033[1;36mSearching for: '{args.question}'...\033[0m")
    
    # We should print the answer if query() or ask() is available, or just results
    results = ragi.search(args.question)
    
    if not results:
        print("\033[1;31mNo relevant documents found.\033[0m")
        return
        
    print("\n\033[1;32mResults:\033[0m")
    for i, res in enumerate(results[:3], 1):
        print(f"\n\033[1;33m[{i}] Source: {res.metadata.get('source', 'Unknown')}\033[0m")
        print(f"{res.content}")

def index_docs(args):
    try:
        from piragi import Ragi
    except ImportError as e:
        print(f"Error importing piragi: {e}")
        return

    print(f"\033[1;36mIndexing {args.source}...\033[0m")
    ragi = Ragi()
    ragi.add(args.source)
    print(f"\033[1;32mSuccessfully indexed {args.source}\033[0m")

def main():
    parser = argparse.ArgumentParser(prog="piragi", description="piragi - zero-config RAG")
    subparsers = parser.add_subparsers(dest="command")
    
    # repl command
    repl_parser = subparsers.add_parser("repl", help="Start interactive REPL")
    
    # ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question against docs")
    ask_parser.add_argument("source", help="Document source (file/dir/URL)")
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.add_argument("--model", default="all-mpnet-base-v2", help="Embedding model")
    ask_parser.add_argument("--backend", default=None, help="Backend (onnx, openvino)")
    
    # index command
    index_parser = subparsers.add_parser("index", help="Index documents")
    index_parser.add_argument("source", help="Document source")
    
    args = parser.parse_args()
    
    if args.command == "repl":
        repl()
    elif args.command == "ask":
        one_shot_ask(args)
    elif args.command == "index":
        index_docs(args)
    elif args.command is None:
        repl()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
