from django.db import transaction
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from chat.models import ChatSession, Message, SelfProfile
from chat.serializers import (
    ChatRequestSerializer,
    ChatResponseSerializer,
    MessageSerializer,
    SelfProfileSerializer,
)
from chat.services.ollama_client import get_chat_response
from chat.services.learning import run_learning_pass, should_trigger_learning
from chat.services.prompt_builder import build_system_prompt


class ChatView(APIView):
    """
    POST /api/chat/

    Body:
      {"message": "...", "session_id": <int|null>}

    - Omit / null session_id to start a new session.
    - Returns the session_id so the client can continue the same conversation.
    """

    def post(self, request):
        serializer = ChatRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        # Resolve session
        session_id = data.get("session_id")
        if session_id:
            try:
                session = ChatSession.objects.get(pk=session_id)
            except ChatSession.DoesNotExist:
                return Response(
                    {"error": "Session not found."},
                    status=status.HTTP_404_NOT_FOUND,
                )
        else:
            session = ChatSession.objects.create()

        # 사용자 메시지·학습·답변을 한 트랜잭션으로 묶는다. Ollama가 실패하면 전부 롤백해서
        # 답 없는 사용자 메시지가 남지 않게 한다(재시도하면 user 메시지가 연속으로 쌓이고
        # 학습 주기 카운트도 어긋났다).
        try:
            with transaction.atomic():
                # Persist user message
                Message.objects.create(
                    session=session,
                    role=Message.ROLE_USER,
                    content=data["message"],
                )

                # Trigger learning before responding (profile informs the reply)
                if should_trigger_learning(session.id):
                    run_learning_pass(session.id)

                # Build conversation history (full session, chronological)
                history = list(
                    Message.objects.filter(session=session).order_by("created_at")
                )
                conversation = [{"role": m.role, "content": m.content} for m in history]

                # Generate reply
                profile = SelfProfile.get_or_create_default()
                reply = get_chat_response(conversation, build_system_prompt(profile))

                # Persist assistant reply
                Message.objects.create(
                    session=session,
                    role=Message.ROLE_ASSISTANT,
                    content=reply,
                )
        except RuntimeError as exc:
            return Response(
                {"error": f"Model unavailable: {exc}"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        return Response(
            ChatResponseSerializer({"session_id": session.id, "reply": reply}).data
        )


class ChatHistoryView(APIView):
    """GET /api/chat/<session_id>/history/"""

    def get(self, request, session_id):
        try:
            session = ChatSession.objects.get(pk=session_id)
        except ChatSession.DoesNotExist:
            return Response(
                {"error": "Session not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        messages = Message.objects.filter(session=session).order_by("created_at")
        return Response(MessageSerializer(messages, many=True).data)


class ProfileView(APIView):
    """GET /api/profile/ — returns current trait state + full observation log."""

    def get(self, request):
        profile = SelfProfile.get_or_create_default()
        return Response(SelfProfileSerializer(profile).data)
