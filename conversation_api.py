
import logging
from collections import defaultdict
from threading import Lock
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from llm_service import generate_medical_reply

# Logging: shows the real Gemini error in the backend terminal.
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["conversation"])

# Temporary in-memory conversation storage for local development.
# Sessions are lost when the server restarts.
_sessions: dict[str, list[dict[str, str]]] = defaultdict(list)
_sessions_lock = Lock()

MAX_HISTORY_MESSAGES = 20


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    history: list[dict[str, str]]
    llm_used: bool


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # Validate the message.
    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=422,
            detail="Message cannot be empty.",
        )

    # Continue an existing session or create a new one.
    session_id = request.session_id or str(uuid4())

    # Add the user message and capture conversation context.
    with _sessions_lock:
        history = _sessions[session_id]
        history.append({
            "role": "user",
            "content": message,
        })

        history[:] = history[-MAX_HISTORY_MESSAGES:]
        context_history = list(history)

    # Ask the LLM service to generate the response.
    try:
        reply, llm_used = generate_medical_reply(context_history)

        if not isinstance(reply, str) or not reply.strip():
            raise ValueError("The LLM returned an empty response.")

        reply = reply.strip()

    except Exception:
        # Keep the API running, but print the actual error to the terminal.
        logger.exception(
            "Chat response generation failed for session %s",
            session_id,
        )

        reply = (
            "I'm unable to generate a response right now. "
            "Please try again later. This chatbot cannot confirm a diagnosis. "
            "If you may be experiencing a medical emergency, seek urgent "
            "medical help instead of waiting for a chatbot."
        )
        llm_used = False

    # Store the assistant response and return the updated history.
    with _sessions_lock:
        history = _sessions[session_id]
        history.append({
            "role": "assistant",
            "content": reply,
        })

        history[:] = history[-MAX_HISTORY_MESSAGES:]
        response_history = list(history)

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        history=response_history,
        llm_used=llm_used,
    )
