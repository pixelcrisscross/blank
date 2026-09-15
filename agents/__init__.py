"""
Shared model factory for ORCA agents.

Central place to configure the LLM backend so every agent uses the
same model and options. Keeps `think=False` applied consistently.
"""
from google.adk.models.lite_llm import LiteLlm


# Prepend to every agent's instruction. qwen3's built-in switch to
# disable the reasoning trace. Works regardless of whether the Ollama
# version honors extra_body={"think": False}.
ORCA_NO_THINK_PREFIX = "/no_think\n\n"


def orca_model() -> LiteLlm:
    """
    Local Ollama + qwen3:8b with thinking mode disabled.

    qwen3 is a reasoning model: by default it emits 1000-3000 tokens
    of internal monologue before every answer. For an ADK workflow
    with multiple agents per turn, that turns a 10-second query into
    a 2-minute one.

    Both `extra_body` (Ollama 0.6+) and the `/no_think` prefix
    (agent instruction) are used to disable it.
    """
    return LiteLlm(
        model="ollama_chat/qwen3:8b",
        extra_body={"think": False},
        timeout=60,
    )