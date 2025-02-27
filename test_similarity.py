import logging
import numpy as np
import torch
import faiss
from database import get_db
from models import Messages, MessageEmbeddings
from embedder import StellaEmbedder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

NUM_NEIGHBORS = 5

crypto_messages = [
    "Bitcoin reached a new all-time high today, surpassing $100,000!",
    "Ethereum's latest upgrade promises faster transactions and lower fees.",
    "Mining Bitcoin is getting harder with every passing day.",
    "Биткоин снова растет, стоит ли покупать сейчас?",
    "Криптовалюты в России: новые законы усложняют майнинг."
]

def build_faiss_index(embeddings, dimension=1024):
    """Создание FAISS индекса для поиска ближайших соседей"""
    index = faiss.IndexFlatL2(dimension)  # Простой индекс L2
    embeddings_np = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings_np)  # Нормализация для косинусного поиска
    index.add(embeddings_np)
    return index

def main():
    db = next(get_db())
    embedder = StellaEmbedder(device=torch.device("cuda"))

    # Генерируем эмбеддинги для тестовых сообщений
    logging.info("Генерация эмбеддингов для тестовых сообщений...")
    test_embeddings = embedder.embed_batch(crypto_messages)

    # Загружаем все эмбеддинги из базы один раз
    logging.info("Загрузка эмбеддингов из базы данных...")
    all_embeddings = db.query(MessageEmbeddings).all()
    if not all_embeddings:
        logging.warning("База message_embeddings пуста!")
        db.close()
        return

    # Преобразуем эмбеддинги и ID в массивы
    embeddings = [emb.embedding for emb in all_embeddings]
    ids = [emb.message_id for emb in all_embeddings]
    logging.info(f"Загружено {len(embeddings)} эмбеддингов из базы.")

    # Создаем FAISS индекс
    logging.info("Создание FAISS индекса...")
    faiss_index = build_faiss_index(embeddings)

    # Поиск ближайших соседей для каждого тестового сообщения
    for msg_text, embedding in zip(crypto_messages, test_embeddings):
        print("=" * 50)
        print(f"Тестовое сообщение: {msg_text}")
        print(f"Размерность эмбеддинга: {len(embedding)}")

        # Поиск с помощью FAISS
        query_vector = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(query_vector)
        distances, indices = faiss_index.search(query_vector, NUM_NEIGHBORS)

        print("Ближайшие соседи из базы:")
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:  # FAISS возвращает -1, если соседей меньше k
                neighbor_id = ids[idx]
                neighbor_msg = db.query(Messages).filter_by(message_id=neighbor_id).first()
                if neighbor_msg:
                    print("-" * 30)
                    print(f"ID: {neighbor_id}, Расстояние: {dist:.4f}")
                    print(f"Текст: {neighbor_msg.text_message}")
                else:
                    print("-" * 30)
                    print(f"ID: {neighbor_id}, Расстояние: {dist:.4f}")
                    print("Текст: Сообщение не найдено в таблице messages.")

        print("=" * 50)

    db.close()

if __name__ == "__main__":
    main()