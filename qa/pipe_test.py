"""
Pipeline QA — runs the full ORCA workflow for a set of queries.

Usage (from the `marine/` directory):
    python qa/pipeline_test.py

Each query runs with a fresh ADK session so state does not bleed
between queries. Takes 30–90 s per query depending on how many tools
fire.
"""
from __future__ import annotations

import asyncio
import sys
import traceback
import uuid
from pathlib import Path

# Ensure `marine/` is on sys.path so `app`, `agents`, `tools` resolve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agent import root_agent  # noqa: E402  (after sys.path shim)


APP = "orca_qa"
USER = "qa_user"


QUERIES = [
    # (query, expected_intent, expected_domains_hint)
    ("what can you do?",                              "meta",               []),
    ("what is the SST off Goa right now?",            "marine_observations", ["ocean"]),
    ("conditions right now at Koramangala",           "weather",            ["weather"]),
    ("where can I fish near Goa tonight?",            "fishery",            ["pfz"]),
    ("are there coral reefs near Lakshadweep?",       "marine_observations", ["coral"]),
    ("is tonight good for bioluminescence in Goa?",   "tourism",            ["biolum"]),
    ("what's the tide at Mumbai tonight?",            "marine_observations", ["tides"]),
    ("are there port warnings off Visakhapatnam?",    "marine_safety",      ["imd"]),
    ("what is the temperature in Koramangala?",       "weather",            ["weather"]),
    ("xyzabcnotaplace",                               None,                 []),
]


async def run_query(
    runner: Runner,
    session_service: InMemorySessionService,
    session_id: str,
    query: str,
) -> dict:
    """Run one query through the workflow. Returns collected events."""
    content = types.Content(role="user", parts=[types.Part(text=query)])
    final = None
    plan = None
    tools_fired: list[str] = []

    async for event in runner.run_async(
        user_id=USER,
        session_id=session_id,
        new_message=content,
    ):
        if not event.content:
            continue

        text = "".join(
            p.text for p in (event.content.parts or []) if p.text
        ).strip()
        if not text:
            continue

        author = event.author or ""

        if author == "orca_planner":
            plan = text
        elif author.startswith("dynamic_data_collection_"):
            domain = author.replace("dynamic_data_collection_", "")
            if domain not in tools_fired:
                tools_fired.append(domain)
        elif author == "orca_final_synthesizer":
            final = text

    return {"plan": plan, "tools": tools_fired, "answer": final}


def _short(text: str | None, limit: int = 400) -> str:
    if not text:
        return "<none>"
    text = text.replace("\n", " ").strip()
    if len(text) > limit:
        return text[:limit] + "..."
    return text


async def main():
    print("=" * 80)
    print("PIPELINE QA")
    print("=" * 80)

    session_service = InMemorySessionService()
    runner = Runner(
        app_name=APP,
        agent=root_agent,
        session_service=session_service,
    )

    total = len(QUERIES)
    passed = 0
    failed = 0
    errored = 0

    for i, (query, expected_intent, expected_domains) in enumerate(QUERIES, 1):
        # Fresh session per query so state does not carry over.
        session_id = f"qa_{i}_{uuid.uuid4().hex[:8]}"
        await session_service.create_session(
            app_name=APP,
            user_id=USER,
            session_id=session_id,
        )

        print()
        print("─" * 80)
        print(f"[{i}/{total}] QUERY: {query}")
        print("─" * 80)

        try:
            result = await run_query(
                runner, session_service, session_id, query,
            )
        except Exception as exc:
            errored += 1
            print(f"ERROR: {exc}")
            traceback.print_exc(limit=3)
            continue

        print(f"TOOLS FIRED: {result['tools']}")
        print(f"ANSWER: {_short(result['answer'])}")

        # Verify expected domains, if specified
        if expected_domains and result["tools"]:
            missing = [d for d in expected_domains if d not in result["tools"]]
            if missing:
                print(f"  ⚠️  expected domains {expected_domains} — "
                      f"missing: {missing}")
                failed += 1
            else:
                print(f"  ✅ all expected domains fired")
                passed += 1
        elif expected_domains:
            print(f"  ⚠️  expected domains {expected_domains} but no tools fired")
            failed += 1
        else:
            passed += 1

    print()
    print("=" * 80)
    print(f"SUMMARY:  passed={passed}  failed={failed}  errored={errored}")
    print("=" * 80)

    return errored == 0


if __name__ == "__main__":
    asyncio.run(main())