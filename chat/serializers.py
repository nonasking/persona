from rest_framework import serializers

from chat.models import Message, SelfProfile, TraitObservation


class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ["id", "role", "content", "created_at"]


class ChatRequestSerializer(serializers.Serializer):
    session_id = serializers.IntegerField(required=False, allow_null=True)
    message = serializers.CharField()


class ChatResponseSerializer(serializers.Serializer):
    session_id = serializers.IntegerField()
    reply = serializers.CharField()


class TraitObservationSerializer(serializers.ModelSerializer):
    class Meta:
        model = TraitObservation
        fields = [
            "id",
            "raw_observations",
            "applied_delta",
            "message_count",
            "learning_rate_used",
            "created_at",
        ]


class SelfProfileSerializer(serializers.ModelSerializer):
    observations = TraitObservationSerializer(many=True, read_only=True)

    class Meta:
        model = SelfProfile
        fields = [
            "id",
            "traits",
            "total_messages_processed",
            "update_count",
            "created_at",
            "updated_at",
            "observations",
        ]
