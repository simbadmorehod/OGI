import logging
from datetime import datetime
import torch
from sqlalchemy import and_, or_, func
from sqlalchemy.orm import aliased
from tqdm import tqdm
import numpy as np
import argparse
import os

# Установка переменной окружения перед импортом torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from database import get_db
from embedder import StellaEmbedder
from models import Messages, MessageEmbeddings

# Конфигурация
EMBEDDING_DIM = 1024  # Для intfloat/multilingual-e5-large-instruct
BATCH_SIZE = 500  # Уменьшено с 1500 для предотвращения ошибки
MAX_TEXT_LENGTH = 4096  # Максимальная длина текста

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

def update_all_embeddings(clear_db: bool = False, retries: int = 3):
    db = next(get_db())
    embedder = StellaEmbedder(device=torch.device("cuda"))

    try:
        if clear_db:
            logging.info("Очистка таблицы message_embeddings...")
            db.query(MessageEmbeddings).delete()
            db.commit()

        ME = aliased(MessageEmbeddings)
        query = db.query(Messages).outerjoin(ME, ME.message_id == Messages.message_id).filter(
            or_(
                ME.message_id == None,
                Messages.local_date_creation > ME.updated_at
            )
        ).order_by(Messages.message_id)

        total_to_process = query.count()
        if total_to_process == 0:
            logging.info("Нет новых или необработанных сообщений для обработки.")
            return

        logging.info(f"Найдено {total_to_process} сообщений для обработки.")
        processed_ids = set()

        with tqdm(total=total_to_process, desc="Обновление эмбеддингов") as pbar:
            offset = 0
            while True:
                messages = query.offset(offset).limit(BATCH_SIZE).all()
                if not messages:
                    break

                texts = []
                valid_messages = []

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

                embeddings = None
                for attempt in range(retries):
                    try:
                        embeddings = embedder.embed_batch(texts)  # Генерация на GPU
                        if embeddings.shape[0] == len(valid_messages):
                            break
                    except Exception as e:
                        logging.error(f"Попытка {attempt + 1}/{retries} провалена: {str(e)}")
                        torch.cuda.empty_cache()
                        if attempt == retries - 1:
                            raise

                batch = []
                for msg, emb in zip(valid_messages, embeddings):
                    if not validate_embedding(emb.tolist()):
                        logging.warning(f"Некорректный эмбеддинг для сообщения {msg.message_id}")
                        continue
                    batch.append({
                        "message_id": msg.message_id,
                        "embedding": emb.tolist(),
                        "updated_at": datetime.now()
                    })
                    processed_ids.add(msg.message_id)

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

                log_gpu_memory()
                torch.cuda.empty_cache()
                offset += len(messages)
                pbar.update(len(valid_messages))

        logging.info(f"Обработано {len(processed_ids)} уникальных сообщений.")

    except Exception as e:
        logging.critical(f"Фатальная ошибка: {str(e)}")
        raise
    finally:
        db.close()
        torch.cuda.empty_cache()
        logging.info("Процесс обновления завершен")

def main():
    parser = argparse.ArgumentParser(description="Обновление эмбеддингов сообщений.")
    parser.add_argument("--clear", action="store_true", help="Очистить таблицу эмбеддингов перед обработкой.")
    args = parser.parse_args()
    update_all_embeddings(clear_db=args.clear)

if __name__ == "__main__":
    main()