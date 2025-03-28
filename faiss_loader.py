import logging
from datetime import datetime, timezone

import numpy as np
from tqdm import tqdm
from database import SessionLocal
from models import MessageEmbeddings
from faiss_manager import FaissManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

CHUNK_SIZE = 10000


def load_embeddings_to_faiss(dimension=1025, index_path="faiss_index.bin"):
    """Загружает эмбеддинги из базы в FAISS порциями с IndexFlatL2, очищая индекс в начале"""
    faiss_manager = FaissManager(dimension=dimension, index_path=index_path)

    db = SessionLocal()
    try:
        # Очищаем индекс перед загрузкой
        faiss_manager.clear_index()
        logging.info("FAISS индекс очищен перед началом загрузки.")

        # Получаем общее количество записей в базе
        total = db.query(MessageEmbeddings).count()
        logging.info(f"Всего записей в базе: {total}")
        if total == 0:
            logging.warning("База message_embeddings пуста!")
            return faiss_manager

        def datetime_to_float(dt):
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
            total_seconds = (dt - epoch).total_seconds()
            return total_seconds

        with tqdm(total=total, desc="Загрузка эмбеддингов в FAISS", unit="rec") as pbar:
            for offset in range(0, total, CHUNK_SIZE):
                chunk = db.query(MessageEmbeddings.message_id, MessageEmbeddings.embedding,
                                 MessageEmbeddings.created_at) \
                    .offset(offset).limit(CHUNK_SIZE).all()
                if not chunk:
                    continue

                chunk_ids = [row.message_id for row in chunk]
                chunk_embeddings = [row.embedding for row in chunk]
                chunk_timestamps = [datetime_to_float(row.created_at) for row in chunk]

                new_embeddings = np.array(chunk_embeddings, dtype=np.float32)

                # Добавляем векторы в индекс, включая временное поле
                faiss_manager.add_vectors(chunk_ids, new_embeddings, chunk_timestamps)

                pbar.update(len(chunk))
                if faiss_manager.index.ntotal > 10000:
                    break

        # Проверяем количество записей в индексе после загрузки
        loaded_total = faiss_manager.index.ntotal
        logging.info(f"Завершена загрузка. В FAISS индексе: {loaded_total} записей.")
        if loaded_total != total:
            logging.warning(f"Несоответствие: в базе {total}, в индексе {loaded_total} записей!")

        return faiss_manager

    except Exception as e:
        logging.error(f"Ошибка при загрузке эмбеддингов в FAISS: {str(e)}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    faiss_manager = load_embeddings_to_faiss()