
"""Main FastAPI application for the AI Medical Chatbot."""

import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from conversation_api import router as conversation_router

app = FastAPI(
    title="AI Medical Chatbot API",
    description="Hybrid RAG-based Medical Question Answering System",
    version="3.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# Conversation endpoints
app.include_router(conversation_router)

# Hybrid RAG endpoints
if os.getenv("ENABLE_RAG", "true").strip().lower() == "true":
    from rag.api import router as rag_router
    app.include_router(rag_router)


@app.get("/")
def root():
    return {
        "message": "AI Medical Chatbot API is running.",
        "architecture": "Hybrid RAG",
    }
