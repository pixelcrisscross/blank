from __future__ import annotations

import asyncio
import os
import re

from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agent import root_agent
from state.schemas import key


load_dotenv()


APP_NAME = "orca_marine_intelligence"
USER_ID = "local_user"
SESSION_ID = "local_session"

VERBOSE = os.getenv("ORCA_VERBOSE", "0") == "1"


# ── Post-processing: strip labels a small model may still emit ────────
_STRIP_PREFIXES = (
    "observation query:",
    "conditions:",
    "assessment:",
    "summary:",
    "answer:",
    "response:",
)


def _strip_label(text: str) -> str:
    """Strip a leading label like 'Observation query:' if present."""
    s = text.lstrip()
    low = s.lower()
    for p in _STRIP_PREFIXES:
        if low.startswith(p):
            s = s[len(p):].lstrip(" \n\t:-")
            low = s.lower()
    return s


def _collapse_whitespace(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _build_map_suffix(state: dict) -> str:
    """
    Return a map footer line if the visualize state contains an OK map.

    This runs in Python, not the model, so the path is always exact.
    """
    viz_key = key("visualize")
    viz = state.get(viz_key)
    if not isinstance(viz, dict):
        return ""
    if viz.get("status") != "OK":
        return ""
    map_path = viz.get("map_path")
    if not map_path:
        return ""
    return f"\n\nI've saved an interactive map to {map_path}"


async def main() -> None:

    print("=" * 70)
    print("🌊 ORCA — Marine Intelligence Platform")
    print("=" * 70)
    print()
    print("Type 'exit' to quit.")
    if VERBOSE:
        print("(verbose mode — every agent event will be printed)")
    print()

    session_service = InMemorySessionService()

    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    runner = Runner(
        app_name=APP_NAME,
        agent=root_agent,
        session_service=session_service,
    )

    while True:

        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            break

        content = types.Content(
            role="user",
            parts=[types.Part(text=user_input)],
        )

        print()

        synthesis: str | None = None
        capability: str | None = None
        error: str | None = None

        try:
            async for event in runner.run_async(
                user_id=USER_ID,
                session_id=SESSION_ID,
                new_message=content,
            ):
                if not event.content:
                    continue

                text = "".join(
                    p.text for p in (event.content.parts or []) if p.text
                ).strip()
                if not text:
                    continue

                author = event.author or "unknown"

                if VERBOSE:
                    print(f"[{author}] {text}\n")
                    continue

                if not event.is_final_response():
                    continue

                if author == "orca_final_synthesizer":
                    synthesis = text
                elif (
                    author == "capability_agent"
                    and "skipping" not in text.lower()
                ):
                    capability = text

        except Exception as exc:
            error = str(exc)

        if VERBOSE:
            print()
            continue

        if error:
            print(f"ORCA: pipeline error — {error}")
            print()
            continue

        # ── Build the user-facing response ────────────────────────────
        if capability:
            response = capability
        elif synthesis:
            response = _strip_label(synthesis)
            response = _collapse_whitespace(response)

            # Append the map footer from state, not from the model.
            session = await session_service.get_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=SESSION_ID,
            )
            if session is not None:
                state = session.state or {}
                response += _build_map_suffix(state)
        else:
            response = None

        if response:
            print(f"ORCA: {response}")
        else:
            print("ORCA: no response produced. Run with ORCA_VERBOSE=1 to debug.")

        print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down...")