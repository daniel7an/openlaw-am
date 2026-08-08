"""Configuration: config.toml for settings, .env for secrets only.

Environment variables win over the file, so a deployment can override any single
value without editing (or forking) config.toml:

    OPENLAW_BASE_URL=http://localhost:11434/v1 uv run python rag.py "..."
"""
import os
import tomllib
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CONFIG_FILE = Path(__file__).parent / "config.toml"
PROMPTS_FILE = Path(__file__).parent / "prompts.toml"
_MISSING = object()


@lru_cache(maxsize=1)
def config() -> dict:
    return tomllib.loads(CONFIG_FILE.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def prompts() -> dict:
    return tomllib.loads(PROMPTS_FILE.read_text(encoding="utf-8"))


def prompt(dotted: str) -> str:
    node = prompts()
    for part in dotted.split("."):
        node = node[part]
    return node


def get(dotted: str, env: str | None = None, default=_MISSING):
    """Read `section.key` from config.toml, letting `env` override it.

    The env value is cast to the type of the config value, so numeric overrides
    like OPENLAW_HYBRID_ALPHA=0.8 don't silently arrive as strings.
    """
    node = config()
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            node = _MISSING
            break
        node = node[part]

    override = os.getenv(env) if env else None
    if override is not None:
        if node is not _MISSING and not isinstance(node, str):
            return type(node)(override)
        return override

    if node is _MISSING:
        if default is _MISSING:
            raise KeyError(f"{dotted} missing from {CONFIG_FILE.name} and no default given")
        return default
    return node


def api_key() -> str | None:
    """Secrets never come from config.toml. Both spellings accepted."""
    return os.getenv("OPENLAW_API_KEY") or os.getenv("OPENROUTER_API_KEY")
