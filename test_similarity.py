import logging
import numpy as np
import torch
from database import get_db  # Предполагается, что у вас есть функция get_db
from models import Messages, MessageEmbeddings  # Ваши модели
from embedder import StellaEmbedder  # Импортируем ваш эмбеддер

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Конфигурация
NUM_NEIGHBORS = 5

# Тестовые сообщения на тему криптовалют (3 на английском, 2 на русском)
crypto_messages = [
    "Bitcoin reached a new all-time high today, surpassing $100,000!",
    "Ethereum's latest upgrade promises faster transactions and lower fees.",
    "Mining Bitcoin is getting harder with every passing day.",
    "Биткоин снова растет, стоит ли покупать сейчас?",
    "Криптовалюты в России: новые законы усложняют майнинг."
]


def cosine_distance(vec1, vec2):
    """Вычисление косинусного расстояния между двумя векторами"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return 1 - (dot_product / (norm1 * norm2)) if norm1 * norm2 != 0 else float('inf')


def find_nearest_neighbors(db, target_embedding, num_neighbors=NUM_NEIGHBORS):
    """Поиск ближайших соседей по косинусному расстоянию в базе"""
    all_embeddings = db.query(MessageEmbeddings).all()
    if not all_embeddings:
        logging.warning("База message_embeddings пуста!")
        return []

    distances = []
    for emb in all_embeddings:
        dist = cosine_distance(np.array(target_embedding), np.array(emb.embedding))
        distances.append((emb.message_id, dist))

    # Сортируем по расстоянию и берем топ-N
    distances.sort(key=lambda x: x[1])
    return distances[:num_neighbors]


def main():
    db = next(get_db())

    # Инициализируем эмбеддер
    embedder = StellaEmbedder(device=torch.device("cuda"))

    # Генерируем эмбеддинги для тестовых сообщений
    logging.info("Генерация эмбеддингов для тестовых сообщений...")
    test_embeddings = embedder.embed_batch(crypto_messages, internal_batch_size=5)

    # Проверяем базу и выводим результаты
    for msg_text, embedding in zip(crypto_messages, test_embeddings):
        print("=" * 50)
        print(f"Тестовое сообщение: {msg_text}")
        print(f"Размерность эмбеддинга: {len(embedding)}")

        # Находим ближайших соседей в базе
        neighbors = find_nearest_neighbors(db, embedding)
        if not neighbors:
            print("Ближайшие соседи не найдены — база пуста.")
            continue

        print("Ближайшие соседи из базы:")
        for neighbor_id, distance in neighbors:
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