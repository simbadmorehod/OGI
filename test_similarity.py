import logging
import numpy as np
import torch
import faiss
from tqdm import tqdm
import gc
from database import get_db
from models import Messages, MessageEmbeddings
from embedder import StellaEmbedder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

NUM_NEIGHBORS = 5
CHUNK_SIZE = 10000
TOP_N = 100

crypto_messages = [
    "Bitcoin reached a new all-time high today, surpassing $100,000!",
    "Ethereum's latest upgrade promises faster transactions and lower fees.",
    "Mining Bitcoin is getting harder with every passing day.",
    "Биткоин снова растет, стоит ли покупать сейчас?",
    "Криптовалюты в России: новые законы усложняют майнинг."
]


def find_top_n_in_chunk(chunk_embeddings, chunk_ids, query_vector, top_n=TOP_N):
    """Поиск топ-N соседей в одной порции с помощью FAISS"""
    index = faiss.IndexFlatL2(1024)
    embeddings_np = np.array(chunk_embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings_np)
    index.add(embeddings_np)

    query_vector_np = np.array([query_vector], dtype=np.float32)
    faiss.normalize_L2(query_vector_np)
    distances, indices = index.search(query_vector_np, top_n)

    return [(chunk_ids[idx], dist) for idx, dist in zip(indices[0], distances[0]) if idx != -1]


def main():
    db = next(get_db())
    embedder = StellaEmbedder(device=torch.device("cuda"))

    # Генерируем эмбеддинги для тестовых сообщений
    logging.info("Генерация эмбеддингов для тестовых сообщений...")
    test_embeddings = embedder.embed_batch(crypto_messages)

    total = db.query(MessageEmbeddings).count()
    logging.info(f"Всего записей в базе: {total}")
    if total == 0:
        logging.warning("База message_embeddings пуста!")
        db.close()
        return

    # Поиск ближайших соседей для каждого тестового сообщения
    for msg_text, embedding in zip(crypto_messages, test_embeddings):
        print("=" * 50)
        print(f"Тестовое сообщение: {msg_text}")
        print(f"Размерность эмбеддинга: {len(embedding)}")

        candidates = []
        with tqdm(total=total, desc="Поиск по порциям", unit="rec") as pbar:
            for offset in range(0, total, CHUNK_SIZE):
                chunk = db.query(MessageEmbeddings).offset(offset).limit(CHUNK_SIZE).all()
                if not chunk:
                    continue

                chunk_embeddings = [emb.embedding for emb in chunk]
                chunk_ids = [emb.message_id for emb in chunk]

                # Находим топ-N в текущей порции
                chunk_candidates = find_top_n_in_chunk(chunk_embeddings, chunk_ids, embedding, TOP_N)
                candidates.extend(chunk_candidates)

                # Очищаем память после использования порции
                del chunk_embeddings, chunk_ids, chunk
                gc.collect()

                pbar.update(CHUNK_SIZE)

        # Сортируем всех кандидатов и выбираем топ-NUM_NEIGHBORS
        candidates.sort(key=lambda x: x[1])
        top_candidates = candidates[:NUM_NEIGHBORS]

        print("Ближайшие соседи из базы:")
        for neighbor_id, distance in top_candidates:
            neighbor_msg = db.query(Messages).filter_by(message_id=neighbor_id).first()
            if neighbor_msg:
                print("-" * 30)
                print(f"ID: {neighbor_id}, Расстояние: {distance:.4f}")
                print(f"Текст: {neighbor_msg.text_message}")
            else:
                print("-" * 30)
                print(f"ID: {neighbor_id}, Расстояние: {distance:.4f}")
                print("Текст: Сообщение не найдено в таблице messages.")
        print("=" * 50)

    db.close()


if __name__ == "__main__":
    main()