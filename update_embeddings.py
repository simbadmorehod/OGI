# update_embeddings.py
import logging
from datetime import datetime
import torch
from sqlalchemy import and_, or_, func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import aliased
from tqdm import tqdm
import numpy as np
from database import get_db
from embedder import StellaEmbedder
from models import Messages, MessageEmbeddings

# Конфигурация
EMBEDDING_DIM = 1024  # Для bge-small-en-v1.5
BATCH_SIZE = 20000  # Оптимальный размер батча для GPU (подберите под вашу видеокарту)
MAX_TEXT_LENGTH = 5000  # Максимальная длина текста

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embedding_update.log'),
        logging.StreamHandler()
    ]
)

def validate_embedding(embedding: list) -> bool:
    """Проверка корректности эмбеддинга"""
    return (
            isinstance(embedding, list) and
            len(embedding) == EMBEDDING_DIM and
            all(isinstance(x, float) for x in embedding)
    )

def log_gpu_memory():
    """Логирование использования памяти GPU"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e6
        reserved = torch.cuda.memory_reserved() / 1e6
        logging.info(f"GPU Memory Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

def update_all_embeddings(retries: int = 3):
    db = next(get_db())
    embedder = StellaEmbedder(device="cuda")  # Используем GPU
    try:
        # Удаляем старые эмбеддинги с неверной размерностью
        db.query(MessageEmbeddings).delete()
        db.commit()

        # Получаем сообщения для обработки
        ME = aliased(MessageEmbeddings)
        query = db.query(Messages).outerjoin(ME, ME.message_id == Messages.message_id).filter(
            or_(
                ME.message_id == None,
                Messages.local_date_creation > ME.updated_at
            )
        ).order_by(Messages.message_id)

        total_to_process = query.count()
        if total_to_process == 0:
            logging.info("Нет новых сообщений для обработки.")
            return

        with tqdm(total=total_to_process, desc="Обновление эмбеддингов") as pbar:
            offset = 0
            while True:
                messages = query.offset(offset).limit(BATCH_SIZE).all()
                if not messages:
                    break

                texts = []
                valid_messages = []

                # Подготовка текстов
                for msg in messages:
                    try:
                        text = (msg.text_message or "").strip()[:MAX_TEXT_LENGTH]
                        if len(text) < 2:
                            raise ValueError("Слишком короткий текст")
                        texts.append(text)
                        valid_messages.append(msg)
                    except Exception as e:
                        logging.warning(f"Пропущено сообщение {msg.message_id}: {str(e)}")
                        continue

                # Генерация эмбеддингов на GPU
                embeddings = None
                for attempt in range(retries):
                    try:
                        embeddings = embedder.embed_batch(texts)  # Генерация на GPU
                        if embeddings.shape[0] == len(valid_messages):
                            break
                    except Exception as e:
                        logging.error(f"Попытка {attempt + 1}/{retries} провалена: {str(e)}")
                        torch.cuda.empty_cache()  # Освобождаем память GPU при ошибке
                        if attempt == retries - 1:
                            raise

                # Валидация и сохранение
                batch = []
                for msg, emb in zip(valid_messages, embeddings.cpu().numpy()):  # Перемещаем данные на CPU
                    if not validate_embedding(emb.tolist()):
                        logging.warning(f"Некорректный эмбеддинг для сообщения {msg.message_id}")
                        continue
                    batch.append({
                        "message_id": msg.message_id,
                        "embedding": emb.tolist(),  # Преобразование numpy массива в список
                        "updated_at": datetime.now()
                    })

                # Пакетная вставка/обновление
                if batch:
                    objects = []
                    for row in batch:
                        obj = db.query(MessageEmbeddings).filter_by(message_id=row["message_id"]).first()
                        if obj:
                            obj.embedding = row["embedding"]
                            obj.updated_at = row["updated_at"]
                        else:
                            obj = MessageEmbeddings(
                                message_id=row["message_id"],
                                embedding=row["embedding"],
                                updated_at=row["updated_at"]
                            )
                        objects.append(obj)
                    db.bulk_save_objects(objects)
                    db.commit()

                # Логирование использования памяти GPU
                log_gpu_memory()

                # Освобождение памяти GPU
                torch.cuda.empty_cache()

                offset += len(messages)
                pbar.update(len(valid_messages))

    except Exception as e:
        logging.critical(f"Фатальная ошибка: {str(e)}")
        raise
    finally:
        db.close()
        torch.cuda.empty_cache()  # Освобождаем память GPU перед завершением
        logging.info("Процесс обновления завершен")

if __name__ == "__main__":
    update_all_embeddings()