"""
Self-learning system — accumulates stable personality traits over time.

Core algorithm: Adaptive Exponential Moving Average (EMA)

  new_score = α × observed + (1 − α) × existing

  α (learning rate) decays linearly from ALPHA_MAX → ALPHA_MIN over the
  first VIRTUAL_SAMPLE_CAP learning passes, then stays at ALPHA_MIN.

Why this avoids the three failure modes of a naive system:

  1. Recency bias       → observation window is 2× trigger interval, so each
                          pass sees more context than just the last batch.
                          And EMA means even a perfect observation moves a
                          mature score by at most α × Δ.

  2. Overfitting        → a single unusual conversation at pass 20 moves a
                          score by at most 0.10 × 1.0 = 0.10 points on a
                          0–1 scale, and only if the observation was at the
                          extreme end.

  3. Inconsistency      → booleans use weighted majority (not flip-on-one),
                          phrases/topics decay gradually rather than replacing,
                          and every change is logged in TraitObservation for audit.
"""

import copy

from django.conf import settings

from chat.models import DEFAULT_TRAITS, Message, SelfProfile, TraitObservation
from chat.services.ollama_client import extract_trait_observations


# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

ALPHA_MAX: float = 0.35        # learning rate when profile is fresh (pass 0)
ALPHA_MIN: float = 0.10        # learning rate floor (mature profile, pass ≥ CAP)
VIRTUAL_SAMPLE_CAP: int = 15   # after this many passes, α stays at ALPHA_MIN

PHRASE_DECAY: float = 0.50     # fraction of α applied as per-pass weight decay
TOPIC_DECAY: float = 0.30      # fraction of α applied as per-pass topic decay
MIN_WEIGHT: float = 0.05       # weights below this are pruned from sets
MAX_PHRASES: int = 20
MAX_TOPICS: int = 15


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def should_trigger_learning(session_id: int) -> bool:
    """Return True when the session has accumulated another full trigger interval."""
    interval = settings.LEARNING_TRIGGER_INTERVAL
    count = Message.objects.filter(
        session_id=session_id, role=Message.ROLE_USER
    ).count()
    return count > 0 and count % interval == 0


def run_learning_pass(session_id: int) -> None:
    """
    Main entry point.  Called after each trigger-interval user message.

    Steps:
      1. Load profile + compute adaptive α
      2. Collect observation window (2× interval, slightly older messages included)
      3. Ask Gemini to extract raw observations
      4. Integrate observations into the profile via EMA / weighted majority
      5. Persist profile + write immutable TraitObservation log entry
    """
    profile = SelfProfile.get_or_create_default()
    alpha = _compute_alpha(profile.update_count)

    # Wider window than the trigger interval → less recency bias per pass
    window = settings.LEARNING_TRIGGER_INTERVAL * 2
    messages = list(
        Message.objects.filter(session_id=session_id, role=Message.ROLE_USER)
        .order_by("-created_at")[:window]
    )
    messages.reverse()  # chronological order for the prompt

    if not messages:
        return

    raw_obs = extract_trait_observations([m.content for m in messages])

    if not raw_obs:
        # Gemini returned unparseable output — skip silently rather than corrupt profile
        return

    delta = _integrate(profile, raw_obs, alpha)

    TraitObservation.objects.create(
        profile=profile,
        raw_observations=raw_obs,
        applied_delta=delta,
        message_count=len(messages),
        learning_rate_used=round(alpha, 4),
    )

    profile.total_messages_processed += len(messages)
    profile.update_count += 1
    profile.save()


# ---------------------------------------------------------------------------
# α schedule
# ---------------------------------------------------------------------------

def _compute_alpha(update_count: int) -> float:
    """
    Linear decay from ALPHA_MAX down to ALPHA_MIN over VIRTUAL_SAMPLE_CAP passes.
    After the cap, α is fixed at ALPHA_MIN so the profile stays stable but
    never completely stops learning.

    Example with defaults (cap=15):
      pass  0 → α = 0.35
      pass  7 → α = 0.225
      pass 15 → α = 0.10
      pass 30 → α = 0.10  (floor)
    """
    t = min(update_count, VIRTUAL_SAMPLE_CAP) / VIRTUAL_SAMPLE_CAP
    return round(ALPHA_MAX - t * (ALPHA_MAX - ALPHA_MIN), 4)


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

def _integrate(profile: SelfProfile, obs: dict, alpha: float) -> dict:
    """
    Merge raw observations into the stable trait store.
    Returns a delta dict that describes every change (used for logging).
    """
    traits = copy.deepcopy(profile.traits or {})
    delta: dict = {}

    # Ensure all canonical keys are present (handles profiles created before
    # a new trait was added to DEFAULT_TRAITS)
    for key, default_val in DEFAULT_TRAITS.items():
        if key not in traits:
            traits[key] = copy.deepcopy(default_val)

    # -- Numeric scores (EMA) ------------------------------------------------
    for field in ("formality", "directness", "humor", "empathy", "verbosity"):
        raw = obs.get(field)
        if not isinstance(raw, (int, float)):
            continue
        clamped = max(0.0, min(1.0, float(raw)))
        old = traits[field]["score"]
        new = round(alpha * clamped + (1 - alpha) * old, 4)
        if old != new:
            delta[field] = {"from": old, "to": new}
        traits[field]["score"] = new
        traits[field]["observations"] = traits[field].get("observations", 0) + 1

    # -- Boolean patterns (EMA on underlying float score, threshold at 0.5) --
    for field in ("uses_lists", "asks_questions", "uses_examples"):
        raw = obs.get(field)
        if not isinstance(raw, bool):
            continue
        old_val = traits[field]["value"]
        old_score = float(traits[field].get("score", 1.0 if old_val else 0.0))
        new_score = round(alpha * (1.0 if raw else 0.0) + (1 - alpha) * old_score, 4)
        new_val = new_score >= 0.5
        if old_val != new_val:
            delta[field] = {"from": old_val, "to": new_val, "score": new_score}
        traits[field]["score"] = new_score
        traits[field]["value"] = new_val
        traits[field]["observations"] = traits[field].get("observations", 0) + 1

    # -- Characteristic phrases (weighted set with decay) --------------------
    if isinstance(obs.get("characteristic_phrases"), list):
        new_phrases = [str(p).strip().lower() for p in obs["characteristic_phrases"] if p]
        traits["characteristic_phrases"] = _merge_phrases(
            traits["characteristic_phrases"], new_phrases, alpha
        )
        delta["characteristic_phrases"] = [p["phrase"] for p in traits["characteristic_phrases"]]

    # -- Topics (weighted set with decay) ------------------------------------
    if isinstance(obs.get("topics"), list):
        new_topics = [str(t).strip().lower() for t in obs["topics"] if t]
        traits["topics"] = _merge_topics(traits["topics"], new_topics, alpha)
        delta["topics"] = list(traits["topics"].keys())

    profile.traits = traits
    return delta


# ---------------------------------------------------------------------------
# Phrase / topic set management
# ---------------------------------------------------------------------------

def _merge_phrases(existing: list[dict], new_phrases: list[str], alpha: float) -> list[dict]:
    """
    Merge observed phrases into the weighted phrase set.

    Per pass:
      - All existing weights decay by (alpha × PHRASE_DECAY)  →  gentle forgetting
      - Observed phrases receive a +alpha boost
      - Newly seen phrases start at alpha × 0.5  (small, must be seen again to survive)
      - Phrases below MIN_WEIGHT are pruned; top MAX_PHRASES are kept
    """
    phrase_map: dict[str, float] = {p["phrase"]: p["weight"] for p in existing}

    # Decay existing
    decay_factor = 1 - alpha * PHRASE_DECAY
    for phrase in list(phrase_map):
        phrase_map[phrase] = round(phrase_map[phrase] * decay_factor, 4)

    # Boost or introduce observed phrases
    for phrase in new_phrases:
        if not phrase:
            continue
        if phrase in phrase_map:
            phrase_map[phrase] = min(1.0, round(phrase_map[phrase] + alpha, 4))
        else:
            phrase_map[phrase] = round(alpha * 0.5, 4)

    # Prune weak entries and return top MAX_PHRASES sorted by weight
    survived = [(p, w) for p, w in phrase_map.items() if w >= MIN_WEIGHT]
    survived.sort(key=lambda x: -x[1])
    return [{"phrase": p, "weight": w} for p, w in survived[:MAX_PHRASES]]


def _merge_topics(existing: dict, new_topics: list[str], alpha: float) -> dict:
    """
    Same decay+boost logic as phrases, but topics are stored as a plain
    {name: weight} dict rather than a list of objects.
    """
    topic_map = dict(existing)

    # Decay all existing topics
    decay_factor = 1 - alpha * TOPIC_DECAY
    for t in list(topic_map):
        topic_map[t] = round(topic_map[t] * decay_factor, 4)

    # Boost or introduce observed topics
    for topic in new_topics:
        if not topic:
            continue
        topic_map[topic] = min(1.0, round(topic_map.get(topic, 0.0) + alpha * 0.5, 4))

    # Prune and cap
    survived = [(t, w) for t, w in topic_map.items() if w >= MIN_WEIGHT]
    survived.sort(key=lambda x: -x[1])
    return dict(survived[:MAX_TOPICS])
