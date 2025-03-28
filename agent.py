import re
from datetime import datetime, timedelta

import numpy as np
import torch
from scipy.spatial.distance import cosine  # Это правильный импорт для расстояния
from models import Messages, MessageEmbeddings
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

    def search_vector(self, query: str, top_k: int = 5):
        try:
            print(f"🔍 Начало поиска для запроса: {query}")
            vector = np.array(self.embedder.embed(query), dtype=np.float32)

            # Поиск с использованием FAISS с учетом времени
            results = self.faiss.search_with_time_constraint(vector, k=top_k * 2, time_constraint_days=1)

            print(f"🔍 Результаты FAISS: {len(results)} записей")

            # Получаем сообщения из базы данных
            message_ids = [result['id'] for result in results if result['score'] < 0.29]

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
            # Приводим вектор к 2D форме
            if vector.ndim == 1:
                vector = vector.reshape(1, -1)
            results = self.faiss.search(vector, top_k)
            print(f"🔍 Результаты FAISS: {len(results)} записей")
            message_ids = [msg['id'] for msg in results if msg.get('score', 0) < 0.29]
            print(f"📌 Отфильтрованные ID: {message_ids}")
            embeddings = self.db.query(MessageEmbeddings).filter(MessageEmbeddings.message_id.in_(message_ids)).all()
            # Вычисление схожести с question_vector
            similarities = []
            for emb in embeddings:
                emb_vector = np.array(emb.embedding, dtype=np.float32)
                similarity = 1 - cosine(question_vector, emb_vector)
                similarities.append((emb.message_id, similarity))

            # Находим самое похожее
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

    def analyze_context(self, messages: list[Messages], question: str) -> str:
        """Анализирует сообщения и формирует ключевые аспекты"""
        context = "\n".join(f"{msg.full_name_sender or msg.username_sender}: {msg.text_message}" for msg in messages)
        prompt = f"""{question}\n\n
        Данные для анализа:\n\n {context}\n\n
        """
        print(2)
        return self.deepseek.answer_question(prompt)

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


            # Генерация похожих вопросов
            # questions = self.deepseek.answer_question(
            #     f"Сгенерируй 10 вопросов, похожих на '{question}', в рамках блокчейн и крипто тематики. Верни только список вопросов, пронумеруй по порядку все 10 вопросов"
            # )
            # # Удаляем теги, если они есть
            # questions = re.sub(r'<[^>]+>', '', questions)
            # print("===========СГЕНЕРИРОВАННЫЕ ПОХОЖИЕ ЗАПРОСЫ==============")
            # print(f"questions: {questions}")
            # # Разбиваем по переносам строк
            # lines = questions.split('\n')
            #
            # # Фильтруем строки, которые начинаются с чисел (1., 2., ..., 10.) и убираем пустые строки
            # question_list = [line.strip() for line in lines if line.strip() and re.match(r'^\d+\.\s', line)]
            #
            # print("===========СГЕНЕРИРОВАННЫЕ ПОХОЖИЕ ЗАПРОСЫ[СПИСОК]==============")
            # print(f"questions: {question_list}")
            # print("===========СГЕНЕРИРОВАННЫЕ ПОХОЖИЕ ЗАПРОСЫ [СПИСОК]==============")
            # print(f"questions_list: {questions}")

            # # Генерация потенциальных ответов
            # answers = self.deepseek.answer_question(
            #     f"Сгенерируй краткие потенциальные ответы на следующие вопросы: {question_list}. Ответы должны быть строго в рамках блокчейн и крипто тематики. В ответе верни только список ответов, пронумеруй по порядку все 10 потенциальных ответов ответов"
            # )
            # answers = re.sub(r'<[^>]+>', '', answers)
            # # Разбиваем по переносам строк
            # lines = answers.split('\n')

            # Фильтруем строки, которые начинаются с чисел (1., 2., ..., 10.) и убираем пустые строки
            # answers = [line.strip() for line in lines if line.strip() and re.match(r'^\d+\.\s', line)]
            # print("===========СГЕНЕРИРОВАННЫЕ ПОХОЖИЕ ОТВЕТЫ==============")
            # print(f"answers: {answers}")
            # print("===========СГЕНЕРИРОВАННЫЕ ПОХОЖИЕ ОТВЕТЫ [СПИСОК]==============")
            # print(f"answers_list: {answers}")
            print("===========НАЙДЕННЫЕ СООБЩЕНИЯ==============")
            # Поиск сообщений по каждому элементу в positive_messages и negative_messages
            messages = []
            for message_list in [positive_messages, negative_messages]:
                for message in message_list:
                    found_messages = self.search_vector(message, top_k=top_k)
                    messages.extend(found_messages)

            # Получение самих сообщений по их ID
            messages = fetch_messages_by_ids(self.db, messages)

            print("+++++++++++++++++++++++++++")
            print(f"messages: {messages}")

            for message in messages:
                print(f"Отправитель: {message.full_name_sender or message.username_sender}\nСоообщение: {message.text_message}\n")
                print("--------------------------")

            # Анализ контекста
            if len(messages) >= 1:
                analysis = self.analyze_context(messages, question)
            else:
                analysis = "Сообщений, похожих на запрос, нет. Ответ будет основан только на общих знаниях."
            analysis = re.sub(r'<[^>]+>', '', analysis)
            print("===========АНАЛИТИКА СООБЩЕНИЙ==============")
            print(f"analysis: {analysis}")

            # # Финальный промт
            # prompt = f"""
            # Ты — эксперт по блокчейну и криптовалютам. Пользователь задал вопрос: \"{question}\".
            #
            # У тебя есть результат анализа из похожих сообщений: {analysis}.
            #
            # Используй этот контекст, чтобы ответить на вопрос пользователя. Если контекст не содержит полезной информации, дай ответ на основе своих знаний.
            #
            # Ответ должен быть на том же языке, что и вопрос (если вопрос на русском, ответь на русском).
            #
            # {"Ответь максимально развёрнуто." if detailed and len(messages) > 1 else "Ответь кратко и по делу."}
            # """
            # response = self.deepseek.answer_question(prompt)
            # response = re.sub(r'<[^>]+>', '', response)
            return analysis if analysis else "Извините, я не смог получить ответ."
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