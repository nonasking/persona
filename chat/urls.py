from django.urls import path

from chat.views import ChatHistoryView, ChatView, ProfileView

urlpatterns = [
    path("chat/", ChatView.as_view(), name="chat"),
    path("chat/<int:session_id>/history/", ChatHistoryView.as_view(), name="chat-history"),
    path("profile/", ProfileView.as_view(), name="profile"),
]
