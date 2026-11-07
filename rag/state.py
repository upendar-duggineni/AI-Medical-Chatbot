"""Per-conversation state for the agent.

Kept in memory only (never written to disk or logged) and cleared after two hours
or when the server restarts.
"""
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

SESSION_TTL = 2 * 60 * 60
MAX_SESSIONS = 500
_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


@dataclass
class SessionState:
    session_id: str
    user_messages: list = field(default_factory=list)  # last few user messages
    symptoms: list = field(default_factory=list)       # the user's own symptom descriptions
    duration: Optional[str] = None
    duration_days: Optional[int] = None
    severity: Optional[str] = None                     # mild | moderate | severe
    age_group: Optional[str] = None                    # infant | child | adult | older_adult
    followups_asked: int = 0
    awaiting: bool = False                             # waiting for the user's follow-up answer
    assessed: bool = False                             # an assessment was already given
    risk: Optional[str] = None                         # self_care | see_doctor | see_doctor_soon | emergency
    flags_shown: set = field(default_factory=set)      # red-flag warnings already shown in this case
    pending_flags: list = field(default_factory=list)  # warnings to attach to the reply being built
    updated: float = field(default_factory=time.time)

    def reset_case(self):
        """Start a fresh symptom case (keeps recent messages for red-flag checks)."""
        self.symptoms, self.duration, self.duration_days = [], None, None
        self.severity, self.age_group, self.risk = None, None, None
        self.followups_asked, self.awaiting, self.assessed = 0, False, False
        self.flags_shown = set()

    def public(self) -> dict:
        return {
            "symptoms": self.symptoms,
            "duration": self.duration,
            "severity": self.severity,
            "age_group": self.age_group,
            "followups_asked": self.followups_asked,
            "risk": self.risk,
        }


_SESSIONS: dict = {}
_LOCK = threading.Lock()


def get_session(session_id: Optional[str] = None) -> SessionState:
    now = time.time()
    with _LOCK:
        for sid in [s for s, v in _SESSIONS.items() if now - v.updated > SESSION_TTL]:
            del _SESSIONS[sid]
        if len(_SESSIONS) >= MAX_SESSIONS:
            oldest = min(_SESSIONS, key=lambda s: _SESSIONS[s].updated)
            del _SESSIONS[oldest]
        if not session_id or not _ID.match(session_id):
            session_id = uuid.uuid4().hex
        state = _SESSIONS.get(session_id)
        if state is None:
            state = SessionState(session_id)
            _SESSIONS[session_id] = state
        state.updated = now
        return state
