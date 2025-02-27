import numpy as np
import torch
from models import Messages
from deepseek_client import DeepSeekClient
from embedder import StellaEmbedder
from faiss_manager import FaissManager
from sqlalchemy.future import select
from sqlalchemy.orm import Session

def fetch_messages_by_ids(db: Session, message_ids: list[int]) -> list[Messages]:
    """Получает объекты Messages по списку message_id"""
    query = select(Messages).where(Messages.message_id.in_(message_ids))
    result = db.execute(query)
    return result.scalars().all()


class CryptoChatAgent:
    def __init__(self, db: Session, faiss_manager: FaissManager,
                 deepseek: DeepSeekClient, embedder: StellaEmbedder):
        self.embedder = embedder  # Инжектируем готовый объект
        self.deepseek = deepseek
        self.faiss = faiss_manager
        self.db = db

    def search_messages(self, query: str, top_k: int):
        try:
            print(f"🔍 Начало поиска для запроса: {query}")
            vector = np.array(self.embedder.embed(query), dtype=np.float32)
            print(f"✅ Вектор сгенерирован. Размерность: {vector.shape}")
            # Приводим вектор к 2D форме
            if vector.ndim == 1:
                vector = vector.reshape(1, -1)
            results = self.faiss.search(vector, top_k)
            print(f"🔍 Результаты FAISS: {len(results)} записей")
# не забыть изменить уровень совпадений до 0.65 или сделать динамическим с уведомлением от этом ИИ
            message_ids = [msg['id'] for msg in results if msg.get('score', 0) > 0.33]
            print(f"📌 Отфильтрованные ID: {message_ids}")

            return fetch_messages_by_ids(self.db, message_ids)
        except Exception as e:
            print(f"🔥 Ошибка при поиске: {str(e)}")
            raise
        finally:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

    def analyze_context(self, messages: list[Messages], question: str) -> str:
        """Анализирует сообщения и формирует ключевые аспекты"""
        context = "\n".join(f"{msg.full_name_sender}: {msg.text_message}" for msg in messages[:20])
        prompt = f"""Структура "full_name_sender:text_message"

        сообщения({len(messages)} вопрос "{question}"):
        {context}
        
        Обрати внимание на всякие аргументы отправителей найди популярные или более значимые события и утверждения"""
        print(2)
        return self.deepseek.answer_question(prompt)

    def answer_question(self, question: str, top_k=10, detailed=False) -> str:
        try:
            """Генерирует ответ на вопрос с учетом контекста сообщений"""
            messages = self.search_messages(question, top_k=top_k)
            print(f"messages: {len(messages)}")
            if len(messages) >= 1:
                analysis = self.analyze_context(messages, question)
            else:
                analysis = "Сообщений похожих под запрос нет, упомяни это при обращении к пользователю и выводам что ответ будет не полный"

            print(f"analysis: {analysis}")

            prompt = f"""На основе следующего контекста, ответь на вопрос:
    
            Контекст:
            {analysis}
    
            Вопрос: {question}
    
            Ответ: """

            if detailed and len(messages) > 1:
                prompt = prompt + " Максимально подробно..."
            else:
                prompt = prompt + " Краткий вывод..."
            print(3)
            response = self.deepseek.answer_question(prompt)
            return response.strip() if response else "Извините, я не смог получить ответ."

        finally:
            # Освобождаем PyTorch GPU / MPS память
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __del__(self):
        """Деструктор для явного освобождения ресурсов"""
        if hasattr(self, 'db'):
            self.db.close()