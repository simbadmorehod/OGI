import torch
from deepseek_client import DeepSeekClient
from embedder import StellaEmbedder
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from database import SessionLocal
from agent import CryptoChatAgent
from faiss_loader import load_embeddings_to_faiss

# Глобальные объекты, загружаются при старте приложения
faiss_manager = load_embeddings_to_faiss()  # Создаем FAISS ОДИН РАЗ
embedder = StellaEmbedder(device=torch.device("cuda"))
deepseek = DeepSeekClient()

app = FastAPI()


class QuestionRequest(BaseModel):
    question: str
    top_k: int
    detailed: bool = False


def get_db():
    """Создание сессии БД для каждого запроса"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/ask/")
def ask_question(request: QuestionRequest, db=Depends(get_db)):
    # Передаём уже загруженные объекты в CryptoChatAgent
    agent = CryptoChatAgent(
        db=db,
        faiss_manager=faiss_manager,
        deepseek=deepseek,
        embedder=embedder
    )
    print(1)
    print("🛑 Запускаем модель перед обработкой запроса")
    # Закрытие DeepSeek
    deepseek.start()
    answer = agent.answer_question(request.question, request.top_k, request.detailed)
    print("🛑 Освобождаем ресурсы, обработке запроса")
    # Закрытие DeepSeek
    deepseek.close()
    return {"response": answer}