import torch

from deepseek_client import DeepSeekClient
from embedder import BGEEmbedder
from fastapi import FastAPI
from pydantic import BaseModel
from database import SessionLocal
from agent import CryptoChatAgent
from faiss_loader import load_embeddings_to_faiss

app = FastAPI()
agent = None  # Глобальный объект для агента
faiss_manager = None


class QuestionRequest(BaseModel):
    question: str
    top_k: int
    detailed: bool = False

def get_db():
    """Зависимость FastAPI для получения сессии"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Убрать ненужные импорты моделей в начале
# Заменить блок инициализации:

@app.on_event("startup")
def startup_event():
    # Инициализация FAISS
    app.state.faiss_manager = load_embeddings_to_faiss()

    # Ленивая загрузка моделей
    app.state.embedder = BGEEmbedder(device="cpu")
    app.state.deepseek = DeepSeekClient()


@app.post("/ask/")
def ask_question(request: QuestionRequest):
    # Создаем агент для каждого запроса
    db = SessionLocal()
    try:
        agent = CryptoChatAgent(
            db=db,
            faiss_manager=app.state.faiss_manager,
            deepseek=app.state.deepseek,
            embedder=app.state.embedder
        )
        return {"response": agent.answer_question(request.question, request.top_k, request.detailed)}
    finally:
        db.close()
        torch.mps.empty_cache()