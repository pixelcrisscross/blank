"""
Language detection service for ORCA.

Detects the language of the user's query so that ORCA can respond in
the same language. Emphasis on Indian regional languages as required by
the competition brief.

Uses `langdetect` (pure Python, offline, MIT licensed).
Falls back gracefully to English if detection fails or the library is
not installed.

Attribution: `langdetect` is a port of Nakatani Shuyo's language-
detection library. MIT license. No network calls.
"""
from __future__ import annotations

from typing import Any


# ── Indian regional languages supported by the platform ──────────────
INDIAN_REGIONAL_LANGUAGES: dict[str, str] = {
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "kn": "Kannada",
    "mr": "Marathi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "or": "Odia",
    "pa": "Punjabi",
    "ur": "Urdu",
    "as": "Assamese",
    "mai": "Maithili",
    "si": "Sinhala",
}

# Languages where ORCA should respond in English regardless
# (most languages ORCA cannot reliably translate to)
_ENGLISH_PASSTHROUGH = {"en", "en-gb", "en-us"}

# Minimum text length for reliable detection
_MIN_DETECT_CHARS = 8


def detect_language(text: str) -> dict[str, Any]:
    """
    Detect the language of a text string.

    Returns
    -------
    dict with keys:
        language_code       : ISO 639-1 code (e.g. "hi", "ta", "en")
        language_name       : Human-readable name (e.g. "Hindi")
        is_indian_regional  : bool — True if one of the 14 supported
                              Indian regional languages
        is_english          : bool
        respond_in_language : bool — True if ORCA should attempt to
                              respond in this language (only for Indian
                              regional languages; English is always True)
        confidence          : "high" | "low" | "unavailable"
        source              : "langdetect" | "fallback"
    """
    text = (text or "").strip()

    # Too short to detect reliably — default to English
    if len(text) < _MIN_DETECT_CHARS:
        return _english_result("text_too_short")

    try:
        from langdetect import detect, detect_langs  # type: ignore[import]
        from langdetect.lang_detect_exception import LangDetectException  # type: ignore[import]

        try:
            probabilities = detect_langs(text)
        except LangDetectException:
            return _english_result("detect_failed")

        if not probabilities:
            return _english_result("no_probabilities")

        top = probabilities[0]
        code = str(top.lang).lower().split("-")[0]  # normalise "zh-cn" → "zh"
        prob = float(top.prob)
        confidence = "high" if prob >= 0.85 else "low"

        is_regional = code in INDIAN_REGIONAL_LANGUAGES
        is_english = code in _ENGLISH_PASSTHROUGH or code == "en"

        return {
            "language_code": code,
            "language_name": INDIAN_REGIONAL_LANGUAGES.get(
                code,
                _LANGUAGE_NAMES.get(code, code.upper()),
            ),
            "is_indian_regional": is_regional,
            "is_english": is_english,
            "respond_in_language": is_regional or is_english,
            "confidence": confidence,
            "probability": round(prob, 3),
            "source": "langdetect",
        }

    except ImportError:
        # langdetect not installed — degrade gracefully
        return _english_result("langdetect_not_installed")
    except Exception:
        return _english_result("unexpected_error")


def _english_result(reason: str) -> dict[str, Any]:
    return {
        "language_code": "en",
        "language_name": "English",
        "is_indian_regional": False,
        "is_english": True,
        "respond_in_language": True,
        "confidence": "unavailable",
        "probability": None,
        "source": "fallback",
        "fallback_reason": reason,
    }


# Common language name lookup for non-Indian languages
_LANGUAGE_NAMES: dict[str, str] = {
    "zh": "Chinese",
    "ar": "Arabic",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "th": "Thai",
    "sw": "Swahili",
}
