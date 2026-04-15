"""
Thin wrapper around Ollama via its OpenAI-compatible API.

Two responsibilities:
  1. get_chat_response()          — persona chat with trimmed history
  2. extract_trait_observations() — structured personality extraction for learning
"""

import json
import re

from openai import OpenAI
from django.conf import settings

# Hard cap on messages sent per request — trims oldest context to reduce load.
# 10 messages ≈ 4 full back-and-forth turns + the current user message.
MAX_HISTORY_MESSAGES: int = 10

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=settings.OLLAMA_BASE_URL,
            api_key="ollama",  # Ollama doesn't validate this — required by the SDK only
        )
    return _client


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

def get_chat_response(conversation: list[dict], system_prompt: str) -> str:
    """
    Send a trimmed conversation history to Ollama and return the reply text.

    Args:
        conversation: list of {"role": "user"|"assistant", "content": str}
        system_prompt: persona prompt built from SelfProfile
    """
    messages = _prepare_messages(conversation, system_prompt)

    try:
        response = _get_client().chat.completions.create(
            model=settings.OLLAMA_MODEL,
            messages=messages,
            max_tokens=512,
        )
        return response.choices[0].message.content
    except Exception as exc:
        raise RuntimeError(f"Ollama chat request failed: {exc}") from exc


def _prepare_messages(conversation: list[dict], system_prompt: str) -> list[dict]:
    """
    Trim to MAX_HISTORY_MESSAGES and prepend the system prompt.

    OpenAI format: system message first, then user/assistant pairs.
    Roles are already "user"/"assistant" — no conversion needed unlike Gemini.
    """
    if len(conversation) > MAX_HISTORY_MESSAGES:
        conversation = conversation[-MAX_HISTORY_MESSAGES:]
        # Ensure the slice starts on a user turn
        if conversation[0]["role"] == "assistant":
            conversation = conversation[1:]

    return [{"role": "system", "content": system_prompt}, *conversation]


# ---------------------------------------------------------------------------
# Personality extraction
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """\
Analyze these messages from one person. Return ONLY a JSON object — no explanation, no markdown.

Messages:
{messages}

JSON schema (all fields required):
{{"formality":0.0-1.0,"directness":0.0-1.0,"humor":0.0-1.0,"empathy":0.0-1.0,\
"verbosity":0.0-1.0,"characteristic_phrases":["up to 5 phrases"],\
"topics":["up to 5 topics"],"uses_lists":true/false,\
"asks_questions":true/false,"uses_examples":true/false,"confidence":0.0-1.0}}"""


def extract_trait_observations(user_messages: list[str]) -> dict:
    """
    Ask Ollama to analyze user messages and return structured personality
    observations as a dict.

    Returns an empty dict on parse failure — callers must handle gracefully.
    """
    numbered = "\n".join(f"{i + 1}. {msg}" for i, msg in enumerate(user_messages))
    prompt = _EXTRACTION_PROMPT.format(messages=numbered)

    try:
        response = _get_client().chat.completions.create(
            model=settings.OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
        )
        return _parse_json_safely(response.choices[0].message.content)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _parse_json_safely(text: str) -> dict:
    """
    Robustly extract a JSON object from the model's response.

    Tries three strategies in order:
      1. Direct json.loads on the stripped response
      2. Strip markdown code fences, then parse
      3. Regex-extract the first {...} block, then parse
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(fenced)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {}
