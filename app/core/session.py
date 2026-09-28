"""
Conversation State and Session Management
Maintains multi-turn context, chat history, detected intent, and metadata across turns.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import uuid
from pydantic import BaseModel, Field


class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str  # "user", "assistant", "system"
    content: str
    intent: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms: Optional[Dict[str, float]] = None


class SessionState(BaseModel):
    session_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    messages: List[Message] = Field(default_factory=list)
    last_intent: Optional[str] = "initial"
    turn_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def add_message(
        self,
        role: str,
        content: str,
        intent: Optional[str] = None,
        latency_ms: Optional[Dict[str, float]] = None
    ) -> Message:
        msg = Message(
            role=role,
            content=content,
            intent=intent,
            latency_ms=latency_ms
        )
        self.messages.append(msg)
        self.updated_at = datetime.now(timezone.utc)
        if intent:
            self.last_intent = intent
        if role == "assistant":
            self.turn_count += 1
        return msg

    def get_history_for_llm(self, max_turns: int = 10) -> List[Dict[str, str]]:
        """Returns the recent conversation history in OpenAI/Groq chat format."""
        # Only take the last max_turns * 2 messages (pairs of user + assistant)
        recent_messages = self.messages[-(max_turns * 2):] if len(self.messages) > (max_turns * 2) else self.messages
        return [{"role": m.role, "content": m.content} for m in recent_messages]

    def clear(self):
        """Resets the conversation history while preserving session_id."""
        self.messages.clear()
        self.last_intent = "initial"
        self.turn_count = 0
        self.updated_at = datetime.now(timezone.utc)


class SessionManager:
    """Thread-safe in-memory session repository."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SessionManager, cls).__new__(cls)
            cls._instance._sessions: Dict[str, SessionState] = {}
        return cls._instance

    def get_or_create_session(self, session_id: Optional[str] = None) -> SessionState:
        if not session_id:
            session_id = str(uuid.uuid4())
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id=session_id)
        return self._sessions[session_id]

    def get_session(self, session_id: str) -> Optional[SessionState]:
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(self) -> List[str]:
        return list(self._sessions.keys())


session_manager = SessionManager()
