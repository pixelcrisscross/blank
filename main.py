from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agent import root_agent


load_dotenv()


APP_NAME = "orca_marine_intelligence"
USER_ID = "local_user"
SESSION_ID = "local_session"

VERBOSE = os.getenv("ORCA_VERBOSE", "0") == "1"


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

        response = capability or synthesis

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