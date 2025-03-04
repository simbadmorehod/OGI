import re

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
            message_ids = [msg['id'] for msg in results if msg.get('score', 0) < 0.29]
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
        context = "\n".join(f"{msg.full_name_sender or msg.username_sender}: {msg.text_message}" for msg in messages[:20])
        prompt = f"""Структура ответа "full_name_sender:text_message/n"
        Данные для анализа: {context}
        
        Найди в сообщениях пользователей значимые события и утверждения для вопроса "{question}")"""
        print(2)
        return self.deepseek.answer_question(prompt)

    def answer_question(self, question: str, top_k=10, detailed=False) -> str:
        try:
            # Генерация похожих вопросов
            questions = self.deepseek.answer_question(
                f"Сгенерируй 10 вопросов, похожих на следующий запрос пользователя, но более развёрнутых и строго в рамках блокчейн и крипто тематики: \"{question}\". В ответе верни только список вопросов, разделённых запятыми, без лишнего текста."
            )
            pattern = r'<think>.*?</think>'
            cleaned_text = re.sub(pattern, '', questions, flags=re.DOTALL)
            questions = cleaned_text.strip()
            print("===========СГЕНЕРИРОВАННЫЕ ПОХОЖИЕ ЗАПРОСЫ==============")
            print(f"questions: {questions}")
            print("===========СГЕНЕРИРОВАННЫЕ ПОХОЖИЕ ЗАПРОСЫ [СПИСОК]==============")
            questions_list = questions.split(", ")
            print(f"questions_list: {questions_list}")

            # Генерация потенциальных ответов
            answers = self.deepseek.answer_question(
                f"Сгенерируй краткие потенциальные ответы на следующие вопросы: {questions}. Ответы должны быть строго в рамках блокчейн и крипто тематики. В ответе верни только список ответов, разделённых запятыми, без лишнего текста."
            )
            cleaned_text = re.sub(pattern, '', answers, flags=re.DOTALL)
            answers = cleaned_text.strip()
            print("===========СГЕНЕРИРОВАННЫЕ ПОХОЖИЕ ОТВЕТЫ==============")
            print(f"answers: {answers}")
            print("===========СГЕНЕРИРОВАННЫЕ ПОХОЖИЕ ОТВЕТЫ [СПИСОК]==============")
            answers_list = answers.split(", ")
            print(f"answers_list: {answers_list}")
            print("===========НАЙДЕННЫЕ СООБЩЕНИЯ==============")
            # Поиск сообщений по каждому ответу
            messages = []
            for answer in answers_list:
                found_messages = self.search_messages(answer, top_k=top_k)
                messages.extend(found_messages)
            print("+++++++++++++++++++++++++++")
            print(f"messages: {messages}")

            # Анализ контекста
            if len(messages) >= 1:
                analysis = self.analyze_context(messages, question)
            else:
                analysis = "Сообщений, похожих на запрос, нет. Ответ будет основан только на общих знаниях."
            cleaned_text = re.sub(pattern, '', analysis, flags=re.DOTALL)
            analysis = cleaned_text.strip()
            print("===========АНАЛИТИКА СООБЩЕНИЙ==============")
            print(f"analysis: {analysis}")

            # Финальный промт
            prompt = f"""
            Ты — эксперт по блокчейну и криптовалютам. Пользователь задал вопрос: \"{question}\".

            У тебя есть результат анализа из похожих сообщений: {analysis}.

            Используй этот контекст, чтобы ответить на вопрос пользователя. Если контекст не содержит полезной информации, дай ответ на основе своих знаний.

            Ответ должен быть на том же языке, что и вопрос (если вопрос на русском, ответь на русском).

            {"Ответь максимально развёрнуто." if detailed and len(messages) > 1 else "Ответь кратко и по делу."}
            """
            response = self.deepseek.answer_question(prompt)
            cleaned_text = re.sub(pattern, '', response, flags=re.DOTALL)
            response = cleaned_text.strip()
            return response.strip() if response else "Извините, я не смог получить ответ."
        except Exception as e:
            return f"Произошла ошибка: {str(e)}"

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