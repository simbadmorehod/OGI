import re
from datetime import datetime, timedelta
import numpy as np
import torch
from scipy.spatial.distance import cosine
from deepseek_client import Dream7BClient
from models import Messages, MessageEmbeddings
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
                 dream_client: Dream7BClient, embedder: StellaEmbedder):
        self.embedder = embedder
        self.dream = dream_client  # Переименовал для ясности
        self.faiss = faiss_manager
        self.db = db
        self.dream.start()  # Инициализируем модель при создании агента

    def search_vector(self, query: str, top_k: int = 5):
        try:
            print(f"🔍 Начало поиска для запроса: {query} top_k:{top_k}")
            vector = np.array(self.embedder.embed(query), dtype=np.float32)
            results = self.faiss.search_with_time_constraint(vector, k=top_k, time_constraint_days=1)
            print(f"🔍 Результаты FAISS: {len(results)} записей")
            message_ids = [result['id'] for result in results if result['score'] < 0.29]  # Исправлено
            print(f"📌 Найдено - search_vector {len(message_ids)} сообщений ")
            return message_ids
        except Exception as e:
            print(f"🔥 Ошибка при поиске search_vector: {str(e)}")
            raise
        finally:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

    def search_messages(self, query: str, question: str, top_k: int):
        try:
            print(f"🔍 Начало поиска для синтетического запроса: {query}")
            vector = np.array(self.embedder.embed(query), dtype=np.float32)
            print(f"🔍 Пользовательский запрос: {question}")
            question_vector = np.array(self.embedder.embed(question), dtype=np.float32)
            print(f"✅ Вектор сгенерирован. Размерность: {vector.shape}")
            if vector.ndim == 1:
                vector = vector.reshape(1, -1)
            results = self.faiss.search(vector, top_k)
            print(f"🔍 Результаты FAISS: {len(results)} записей")
            message_ids = [msg['id'] for msg in results if msg.get('score', 0) < 0.29]
            print(f"📌 Отфильтрованные ID: {message_ids}")
            embeddings = self.db.query(MessageEmbeddings).filter(MessageEmbeddings.message_id.in_(message_ids)).all()
            similarities = []
            for emb in embeddings:
                emb_vector = np.array(emb.embedding, dtype=np.float32)
                similarity = 1 - cosine(question_vector, emb_vector)
                similarities.append((emb.message_id, similarity))
            most_similar = max(similarities, key=lambda x: x[1])
            message_id = most_similar[0]
            return message_id
        except Exception as e:
            print(f"🔥 Ошибка при поиске: {str(e)}")
            raise
        finally:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

    def fetch_recent_messages(self):
        """Получает все сообщения за последние сутки из таблицы message_embeddings."""
        time_threshold = datetime.utcnow() - timedelta(days=1)
        query = select(
            MessageEmbeddings.message_id,
            MessageEmbeddings.embedding
        ).where(
            MessageEmbeddings.created_at >= time_threshold
        )
        result = self.db.execute(query).fetchall()
        recent_messages = [{"message_id": row[0], "embedding": np.array(row[1], dtype=np.float32)} for row in result]
        print(f"📌 Найдено {len(recent_messages)} сообщений за последние сутки")
        return recent_messages

    def find_similar_messages(self, query_text: str, recent_messages: list, threshold: float = 0.29):
        """Ищет похожие сообщения по косинусному расстоянию среди recent_messages."""
        query_embedding = np.array(self.embedder.embed(query_text), dtype=np.float32)
        embeddings = np.stack([msg["embedding"] for msg in recent_messages])
        dot_product = np.dot(embeddings, query_embedding)
        norm_query = np.linalg.norm(query_embedding)
        norm_embeddings = np.linalg.norm(embeddings, axis=1)
        cosine_similarities = dot_product / (norm_query * norm_embeddings)
        cosine_distances = 1 - cosine_similarities
        min_distance_idx = np.argmin(cosine_distances)
        min_distance = cosine_distances[min_distance_idx]
        if min_distance < threshold:
            return recent_messages[min_distance_idx]["message_id"]
        return None

    def analyze_context(self, messages: list[Messages], question: str) -> str:
        """Анализирует сообщения и формирует ключевые аспекты"""
        context = "\n".join(f"{msg.full_name_sender or msg.username_sender}: {msg.text_message}" for msg in messages)
        prompt = f"""
        Вы — аналитик, который составляет отчёт на основе сообщений. 
        Ваша задача: проанализировать предоставленные данные и составить отчёт о позитивных и негативных событиях, а также интересных дискуссиях, связанных с запросом "{question}".

        Формат ответа:
        1. Позитивные события: [список событий]
        2. Негативные события: [список событий]
        3. Интересные дискуссии: [список тем или обсуждений]

        Данные для анализа:
        {context}

        Составьте отчёт:
        """
        print("📝 Промпт для анализа:\n", prompt)
        response = self.dream.answer_question(prompt)
        response = re.sub(r'<[^>]+>', '', response)
        return response

    def answer_question(self, question: str, top_k=10, detailed=False) -> str:
        try:
            positive_messages = [
                "Заебись, цена взлетела!",
                "Круто, наконец-то рост!",
                "Отлично, моя ставка сыграла!",
                "Блин, как классно, что проект взлетел!",
                "Вау, цена поднялась, заебись день!",
                "Супер, мой портфель вырос за ночь!",
                "Класс, новый ATH, молодцы!",
                "Пиздец как круто, что добавили новые токены!",
                "Снова в зеленой зоне, заебись!",
                "Прорвал сопротивление, заебись новость!"
            ]
            negative_messages = [
                "Пиздец, цена опять упала!",
                "Блять, мой стейблкоин потерял пег!",
                "Хуета какая-то, проект снова рухнул!",
                "Пиздец, сколько можно терять?!",
                "Блять, опять дамп на рынке, все летит вниз!",
                "Пиздец, какой-то скам проект обрушил цену!",
                "Хуета полная, мой портфель просел за день!",
                "Блять, опять проблемы с выводом средств!",
                "Пиздец, какой-то кит манипулирует рынком!",
                "Хуета, опять слухи о запрете!"
            ]

            print("===========НАЙДЕННЫЕ СООБЩЕНИЯ==============")
            recent_messages = self.fetch_recent_messages()
            messages = []
            for message_list in [positive_messages, negative_messages]:
                for message in message_list:
                    similar_message_id = self.find_similar_messages(message, recent_messages, threshold=0.29)
                    if similar_message_id:
                        messages.append(similar_message_id)
            unique_messages = list(set(messages))
            print(f"📌 Найдено {len(messages)} сообщений, после удаления дубликатов: {len(unique_messages)}")
            messages = fetch_messages_by_ids(self.db, unique_messages)
            for message in messages:
                print(f"Отправитель[{message.date_creation}]: {message.full_name_sender or message.username_sender}\nСообщение: {message.text_message}\n")
                print("--------------------------")

            if len(messages) >= 1:
                analysis = self.analyze_context(messages, question)
            else:
                analysis = "Сообщений, похожих на запрос, нет. Ответ будет основан только на общих знаниях."
            analysis = re.sub(r'<[^>]+>', '', analysis)
            print("===========АНАЛИТИКА СООБЩЕНИЙ==============")
            print(f"analysis: {analysis}")

            prompt = f"""
            Ты — эксперт по блокчейну и криптовалютам. Пользователь задал вопрос: \"{question}\".

            У тебя есть результат анализа из похожих сообщений: {analysis}.

            Используй этот контекст, чтобы ответить на вопрос пользователя. Если контекст не содержит полезной информации, дай ответ на основе своих знаний.

            Ответ должен быть на том же языке, что и вопрос (если вопрос на русском, ответь на русском).

            {"Ответь максимально развёрнуто." if detailed and len(messages) > 1 else "Ответь кратко и по делу."}
            """
            response = self.dream.answer_question(prompt)
            response = re.sub(r'<[^>]+>', '', response)
            return response if response else "Извините, я не смог получить ответ."
        except Exception as e:
            return f"Произошла ошибка: {str(e)}"
        finally:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __del__(self):
        """Деструктор для освобождения ресурсов"""
        if hasattr(self, 'db'):
            self.db.close()
        if hasattr(self, 'dream'):
            self.dream.close()