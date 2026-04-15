"""
Converts a SelfProfile's structured trait dict into a compact system prompt.

Each instruction is kept to a short phrase — longer sentences carry no extra
signal for the model and waste input tokens on every chat request.
"""

from chat.models import SelfProfile


def build_system_prompt(profile: SelfProfile) -> str:
    """
    Build the persona system prompt from structured traits.

    Only emits instructions for traits that have drifted meaningfully from the
    neutral midpoint (0.5) so the prompt stays minimal and non-contradictory.
    """
    traits = profile.traits
    if not traits:
        return "You are a helpful assistant."

    parts = ["Respond in the user's voice and style."]

    formality = _score(traits, "formality")
    if formality < 0.30:
        parts.append("Tone: very casual.")
    elif formality > 0.70:
        parts.append("Tone: formal and professional.")

    directness = _score(traits, "directness")
    if directness > 0.70:
        parts.append("Be direct; skip filler.")
    elif directness < 0.30:
        parts.append("Be nuanced; avoid bluntness.")

    humor = _score(traits, "humor")
    if humor > 0.65:
        parts.append("Light humor welcome.")
    elif humor < 0.25:
        parts.append("Tone: serious.")

    empathy = _score(traits, "empathy")
    if empathy > 0.70:
        parts.append("Show warmth and empathy.")
    elif empathy < 0.30:
        parts.append("Be analytical.")

    verbosity = _score(traits, "verbosity")
    if verbosity < 0.30:
        parts.append("Be concise.")
    elif verbosity > 0.70:
        parts.append("Elaborate with context.")

    phrases = [p["phrase"] for p in (traits.get("characteristic_phrases") or [])[:5]]
    if phrases:
        quoted = ", ".join(f'"{p}"' for p in phrases)
        parts.append(f"Use phrases: {quoted}.")

    topics = list((traits.get("topics") or {}).keys())[:5]
    if topics:
        parts.append(f"Comfortable topics: {', '.join(topics)}.")

    if _bool(traits, "uses_lists"):
        parts.append("Use bullet points for lists.")

    if _bool(traits, "asks_questions"):
        parts.append("Ask follow-up questions.")

    if _bool(traits, "uses_examples"):
        parts.append("Use concrete examples.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score(traits: dict, key: str) -> float:
    return float(traits.get(key, {}).get("score", 0.5))


def _bool(traits: dict, key: str) -> bool:
    return bool(traits.get(key, {}).get("value", False))
