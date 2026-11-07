import logging
import os
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .pipeline import ask
from .agent import run_turn
from .search import index_info

logger = logging.getLogger("rag.api")
router = APIRouter(prefix="/api", tags=["ask"])


class AgentChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    session_id: Optional[str] = Field(default=None, max_length=100)


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)


class Source(BaseModel):
    n: int
    title: str
    url: str


class AskResponse(BaseModel):
    mode: str  # emergency | dose_refused | out_of_scope | no_source | grounded | extractive
    answer: str
    sources: List[Source] = Field(default_factory=list)
    attribution: str
    disclaimer: str
    # Fixed warnings from safety.check(). Without this field FastAPI's response_model silently
    # strips "red_flags" from whatever pipeline.ask() returns.
    red_flags: List[str] = Field(default_factory=list)
    debug: Optional[List[dict]] = None


@router.post("/agent/chat")
def agent_chat(request: AgentChatRequest):
    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty",
        )

    try:
        return run_turn(
            session_id=request.session_id,
            message=message,
        )
    except HTTPException:
        raise
    except FileNotFoundError:
        logger.exception("RAG knowledge base is unavailable")
        raise HTTPException(
            status_code=503,
            detail="The knowledge base is not set up yet.",
        )
    except ValueError:
        logger.exception("RAG knowledge base file is invalid")
        raise HTTPException(
            status_code=503,
            detail="The knowledge base file is invalid. Check the server log.",
        )
    except Exception:
        logger.exception("Agent chat request failed")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong. Please try again.",
        )


@router.get("/ask/status")
def status():
    try:
        return {
            "ready": True,
            **index_info(),
            "llm_configured": bool(os.getenv("GEMINI_API_KEY")),
        }
    except FileNotFoundError as exc:
        return {"ready": False, "detail": str(exc)}
    except Exception:
        # Keep internal details in the server log, not in the API response.
        logger.exception("RAG status check failed")
        raise HTTPException(
            status_code=500,
            detail="Unable to check chatbot status right now.",
        )


@router.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    try:
        return ask(question)
    except HTTPException:
        raise
    except FileNotFoundError:
        logger.exception("RAG knowledge base is unavailable")
        raise HTTPException(
            status_code=503,
            detail=(
                "The knowledge base is not set up yet. Put "
                "medlineplus_topics.jsonl in data/processed/ "
                "(or set RAG_DATA_FILE)."
            ),
        )
    except ValueError:
        logger.exception("RAG knowledge base file is invalid")
        raise HTTPException(
            status_code=503,
            detail="The knowledge base file is invalid. Check the server log.",
        )
    except Exception:
        # The full traceback is printed in the uvicorn terminal.
        logger.exception("ask failed")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong. Please try again.",
        )
