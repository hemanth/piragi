from src.piragi.core import Ragi
# I'll just write a script to rewrite core.py so I don't mess up multiple replacements
import sys
import re

content = open("src/piragi/core.py").read()
# Replace init args processing
old_init = """        # Initialize config
        cfg = config or {}

        # Store config for later use
        self._config = cfg"""

new_init = """        # Initialize config
        from .config import RagiConfig
        if isinstance(config, RagiConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = RagiConfig(**config)
        else:
            self.config = RagiConfig()

        cfg = config or {}

        # Store config for later use
        self._config = cfg"""
content = content.replace(old_init, new_init)

# Replace chunk strategy
content = content.replace('chunk_cfg = cfg.get("chunk", {})\n        chunk_strategy = chunk_cfg.get("strategy", "fixed")', 'chunk_strategy = self.config.chunk.strategy')
content = content.replace('similarity_threshold=chunk_cfg.get("similarity_threshold", 0.5)', 'similarity_threshold=self.config.chunk.similarity_threshold')
content = content.replace('min_chunk_size=chunk_cfg.get("min_size", 100)', 'min_chunk_size=self.config.chunk.min_size')
content = content.replace('max_chunk_size=chunk_cfg.get("max_size", 2000)', 'max_chunk_size=self.config.chunk.max_size')

content = content.replace('llm_cfg = cfg.get("llm", {})', 'llm_cfg = self.config.llm')
# But wait, later we have `llm_cfg = cfg.get("llm", {})` again. So I should just replace `.get()` with dots.
# Let's do it carefully with sed or python
