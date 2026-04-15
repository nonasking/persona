from django.contrib import admin

from chat.models import ChatSession, Message, SelfProfile, TraitObservation


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ["id", "created_at", "updated_at"]


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ["id", "session", "role", "content_preview", "created_at"]
    list_filter = ["role"]

    def content_preview(self, obj):
        return obj.content[:80]
    content_preview.short_description = "Content"


@admin.register(SelfProfile)
class SelfProfileAdmin(admin.ModelAdmin):
    list_display = ["id", "update_count", "total_messages_processed", "updated_at"]
    readonly_fields = ["traits", "update_count", "total_messages_processed", "created_at", "updated_at"]


@admin.register(TraitObservation)
class TraitObservationAdmin(admin.ModelAdmin):
    list_display = ["id", "profile", "message_count", "learning_rate_used", "created_at"]
    readonly_fields = ["profile", "raw_observations", "applied_delta", "message_count", "learning_rate_used", "created_at"]
