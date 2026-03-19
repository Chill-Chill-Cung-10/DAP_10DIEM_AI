"""FastAPI entrypoint."""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

from Agent import app, test_pg_connection, append_chat_history

api = FastAPI(title="DAP Chatbot API")


class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[dict]] = []


class ChatResponse(BaseModel):
    answer: str
    chat_history: List[dict]


@api.get("/test")
def test():
    db_ok = test_pg_connection()
    return {
        "status": "ok",
        "database": "connected" if db_ok else "unavailable",
    }


@api.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    out = app.invoke({
        "question": request.question,
        "chat_history": request.chat_history,
    })
    answer      = out.get("answer", "(no answer)")
    chat_history = append_chat_history(request.chat_history, request.question, answer)
    return ChatResponse(answer=answer, chat_history=chat_history)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:api", host="0.0.0.0", port=3400, reload=True)