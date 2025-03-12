import logging
from datetime import datetime
import torch
from sqlalchemy import and_, or_, func
from sqlalchemy.orm import aliased
from tqdm import tqdm
import numpy as np
import argparse
import os
import json

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from database import get_db
from embedder import StellaEmbedder
from models import Messages, MessageEmbeddings

# Конфигурация
EMBEDDING_DIM = 1024
BATCH_SIZE = 500
MAX_TEXT_LENGTH = 4096
CHECKPOINT_FILE = "embedding_checkpoint.json"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embedding_update.log'),
        logging.StreamHandler()
    ]
)


def validate_embedding(embedding: list) -> bool:
    return (
            isinstance(embedding, list) and
            len(embedding) == EMBEDDING_DIM and
            all(isinstance(x, float) for x in embedding)
    )


def log_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e6
        reserved = torch.cuda.memory_reserved() / 1e6
        logging.info(f"GPU Memory Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)
            return checkpoint.get('last_offset', 0)
    return 0


def save_checkpoint(offset):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({'last_offset': offset}, f)


def get_query(db, mode: str):
    """Получение запроса в зависимости от режима"""
    ME = aliased(MessageEmbeddings)

    if mode == "incremental":
        # Режим добавления новых сообщений
        query = db.query(Messages).outerjoin(
            ME, ME.message_id == Messages.message_id
        ).filter(
            or_(
                ME.message_id == None,  # Новые сообщения без эмбеддингов
                Messages.local_date_creation > ME.updated_at  # Обновленные сообщения
            )
        ).order_by(Messages.message_id)
    else:  # full mode
        # Полная переобработка всех сообщений
        query = db.query(Messages).order_by(Messages.message_id)

    return query


def update_embeddings(mode: str = "incremental", retries: int = 3):
    db = next(get_db())
    embedder = StellaEmbedder(device=torch.device("cuda"))

    try:
        # Очистка базы только в режиме полной переработки
        if mode == "full":
            logging.info("Очистка таблицы message_embeddings...")
            db.query(MessageEmbeddings).delete()
            db.commit()
            if os.path.exists(CHECKPOINT_FILE):
                os.remove(CHECKPOINT_FILE)

        start_offset = load_checkpoint()
        logging.info(f"Начинаем с offset: {start_offset}")

        # Получаем соответствующий запрос
        query = get_query(db, mode)
        total_to_process = query.count()

        if total_to_process == 0:
            logging.info("Нет сообщений для обработки в выбранном режиме.")
            return

        processed_count = query.offset(0).limit(start_offset).count()
        remaining_to_process = total_to_process - processed_count
        logging.info(f"Режим: {mode}, Всего сообщений: {total_to_process}, Осталось: {remaining_to_process}")

        with tqdm(total=remaining_to_process, desc=f"Обновление эмбеддингов ({mode})", initial=0) as pbar:
            offset = start_offset
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
                        embeddings = embedder.embed_batch(texts)
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
                save_checkpoint(offset)
                pbar.update(len(valid_messages))

        logging.info(f"Обработка завершена в режиме {mode}, последний offset: {offset}")
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)

    except Exception as e:
        logging.critical(f"Фатальная ошибка: {str(e)}")
        raise
    finally:
        db.close()
        torch.cuda.empty_cache()
        logging.info(f"Процесс обновления завершен в режиме {mode}")


def main():
    parser = argparse.ArgumentParser(description="Обновление эмбеддингов сообщений.")
    parser.add_argument(
        "--mode",
        choices=["incremental", "full"],
        default="incremental",
        help="Режим работы: incremental (только новые/обновленные) или full (полная переобработка)"
    )
    args = parser.parse_args()
    update_embeddings(mode=args.mode)


if __name__ == "__main__":
    main()