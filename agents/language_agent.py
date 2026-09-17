"""
Language Detection Agent for ORCA.

Runs as the first step in the workflow (before the planner) to detect
the user's language and store it in session state. The synthesis agent
reads this to respond in the user's language.

Competition brief requirement:
  "Automatically identifying the language of the user's query and
   responding in the same language, with emphasis on supporting Indian
   regional languages."
"""
from __future__ import annotations

import asyncio

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

from tools.language_service import detect_language


_LANGUAGE_CODE = "orca_language"
_LANGUAGE_NAMES_NATIVE: dict[str, str] = {
    "hi": "हिन्दी",
    "ta": "தமிழ்",
    "te": "తెలుగు",
    "ml": "മലയാളം",
    "kn": "ಕನ್ನಡ",
    "mr": "मराठी",
    "bn": "বাংলা",
    "gu": "ગુજરાતી",
    "or": "ଓଡ଼ିଆ",
    "pa": "ਪੰਜਾਬੀ",
    "ur": "اردو",
}


class LanguageDetectionAgent(BaseAgent):
    """
    Detects the user's language from the latest message and stores:

        session.state["orca_language"] = {
            "language_code":      "hi",
            "language_name":      "Hindi",
            "is_indian_regional": True,
            "respond_in_language": True,
        }

    Always yields exactly one Event so the SequentialAgent can continue.
    This agent never blocks the pipeline — if detection fails it
    defaults to English and continues.
    """

    async def _run_async_impl(self, ctx: InvocationContext):
        # Extract the latest user message text
        user_text = ""
        try:
            history = ctx.session.events or []
            for event in reversed(history):
                if event.content and event.content.role == "user":
                    parts = event.content.parts or []
                    user_text = " ".join(
                        p.text for p in parts if p.text
                    ).strip()
                    if user_text:
                        break
        except Exception:
            pass

        # Detect language (runs in thread to avoid blocking the event loop)
        if user_text:
            lang = await asyncio.to_thread(detect_language, user_text)
        else:
            lang = {
                "language_code": "en",
                "language_name": "English",
                "is_indian_regional": False,
                "is_english": True,
                "respond_in_language": True,
                "confidence": "unavailable",
                "source": "fallback",
                "fallback_reason": "no_user_text",
            }

        # Store in session state
        ctx.session.state[_LANGUAGE_CODE] = lang

        # Human-readable status line
        if lang.get("is_indian_regional"):
            native = _LANGUAGE_NAMES_NATIVE.get(
                lang["language_code"], lang["language_name"]
            )
            status = (
                f"Language detected: {lang['language_name']} ({native}) "
                f"[code={lang['language_code']}, "
                f"confidence={lang.get('confidence')}]. "
                f"ORCA will respond in {lang['language_name']}."
            )
        elif lang.get("language_code") == "en":
            status = f"Language: English — standard response."
        else:
            status = (
                f"Language detected: {lang['language_name']} "
                f"[code={lang['language_code']}]. "
                f"ORCA will respond in English (regional language not "
                f"in supported set)."
            )

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part(text=status)],
            ),
        )
