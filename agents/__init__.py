"""
Shared model factory for ORCA agents.

Uses llama3.1:8b — a non-reasoning model with native tool calling.
No thinking mode exists, so no tokens are wasted on internal
monologue. Fits entirely in 6 GB VRAM with room for KV cache.
"""
from google.adk.models.lite_llm import LiteLlm


# Kept for backwards compatibility with agents that import it.
# llama3.2 has no reasoning trace, so this is now empty.
ORCA_NO_THINK_PREFIX = ""


def orca_model() -> LiteLlm:
    """
    Local Ollama + llama3.1:8b.

    llama3.2  :3b:
      - No internal monologue (unlike qwen3)
      - Native tool calling support
      - ~2.0 GB weights, fits 100% on GPU
      - Reliable JSON output

    Timeout 180 s — even a slow multi-turn reasoning call should
    finish well under this on a 3B model.
    """
    return LiteLlm(
        model="ollama_chat/llama3.1:8b",
        timeout=180,
    )