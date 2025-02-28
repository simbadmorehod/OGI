import torch
from deepseek_client import DeepSeekClient
from embedder import StellaEmbedder
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from database import SessionLocal
from agent import CryptoChatAgent
from faiss_loader import load_embeddings_to_faiss

# Проверка CUDA
if not torch.cuda.is_available():
    raise RuntimeError("CUDA недоступен, сервер не может запуститься.")

# Глобальные объекты
# faiss_manager = load_embeddings_to_faiss()
faiss_manager = ''
embedder = StellaEmbedder(device=torch.device("cuda"))
deepseek = DeepSeekClient()
deepseek.start()  # Загружаем модель один раз при старте

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str
    top_k: int
    detailed: bool = False

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/ask/")
def ask_question(request: QuestionRequest, db=Depends(get_db)):
    agent = CryptoChatAgent(
        db=db,
        faiss_manager=faiss_manager,
        deepseek=deepseek,
        embedder=embedder
    )
    print(1)
    print("🛑 Обработка запроса")
    answer = agent.answer_question(request.question, request.top_k, request.detailed)
    return {"response": answer}

@app.on_event("shutdown")
def shutdown_event():
    print("🛑 Освобождаем ресурсы при завершении сервера")
    deepseek.close()  # Освобождаем модель при выключении сервера