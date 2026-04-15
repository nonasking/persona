import copy

from django.db import models


# ---------------------------------------------------------------------------
# Canonical default trait schema — every SelfProfile starts from this.
# Stored in JSONField; never mutated in-place.
# ---------------------------------------------------------------------------
DEFAULT_TRAITS: dict = {
    # Numeric scores (0.0 – 1.0); EMA-blended on each learning pass
    "formality":      {"score": 0.5, "observations": 0},  # 0=casual, 1=formal
    "directness":     {"score": 0.5, "observations": 0},  # 0=indirect, 1=blunt
    "humor":          {"score": 0.5, "observations": 0},  # 0=serious, 1=humorous
    "empathy":        {"score": 0.5, "observations": 0},  # 0=analytical, 1=warm
    "verbosity":      {"score": 0.5, "observations": 0},  # 0=concise, 1=verbose
    # Boolean patterns; EMA on a 0-1 float score, thresholded at 0.5
    # score=0.0 means "never observed", score=1.0 means "always observed"
    "uses_lists":     {"value": False, "score": 0.0, "observations": 0},
    "asks_questions": {"value": False, "score": 0.0, "observations": 0},
    "uses_examples":  {"value": False, "score": 0.0, "observations": 0},
    # Weighted sets; phrases/topics decay each pass, observed ones get boosted
    "characteristic_phrases": [],   # [{"phrase": str, "weight": float}, ...]
    "topics":                 {},   # {"topic_name": weight, ...}
}


class ChatSession(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"Session {self.pk}"


class Message(models.Model):
    ROLE_USER = "user"
    ROLE_ASSISTANT = "assistant"
    ROLE_CHOICES = [(ROLE_USER, "User"), (ROLE_ASSISTANT, "Assistant")]

    session = models.ForeignKey(
        ChatSession, on_delete=models.CASCADE, related_name="messages"
    )
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self) -> str:
        return f"[{self.role}] {self.content[:60]}"


class SelfProfile(models.Model):
    """
    Singleton personality model.
    Use SelfProfile.get_or_create_default() everywhere — never query directly.
    """
    traits = models.JSONField(default=dict)
    total_messages_processed = models.IntegerField(default=0)
    update_count = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    @classmethod
    def get_or_create_default(cls) -> "SelfProfile":
        profile, _ = cls.objects.get_or_create(
            pk=1, defaults={"traits": copy.deepcopy(DEFAULT_TRAITS)}
        )
        return profile

    def __str__(self) -> str:
        return f"SelfProfile (updates={self.update_count})"


class TraitObservation(models.Model):
    """
    Immutable audit log — one row per learning pass.
    Records what the LLM extracted, what actually changed, and the learning rate used.
    Never updated after creation.
    """
    profile = models.ForeignKey(
        SelfProfile, on_delete=models.CASCADE, related_name="observations"
    )
    raw_observations = models.JSONField()   # what the LLM returned
    applied_delta = models.JSONField()      # what changed in the profile
    message_count = models.IntegerField()  # how many messages were analyzed
    learning_rate_used = models.FloatField()

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"Observation at {self.created_at:%Y-%m-%d %H:%M} (α={self.learning_rate_used})"
