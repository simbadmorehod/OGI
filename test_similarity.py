import numpy as np
import faiss
from embedder import BGEEmbedder

# Инициализация эмбеддера
embedder = BGEEmbedder(device="cpu")  # Используйте "cuda", если есть GPU

# Тестовые сообщения
message1 = "ETH растет как на стероидах, скоро будет $5000!"
message2 = "Ethereum взлетает, цена может дойти до 5000 долларов."

# Генерация эмбеддингов
embedding1 = embedder.embed(message1)
embedding2 = embedder.embed(message2)

# Проверка размерности
print(f"Размерность эмбеддинга 1: {embedding1.shape}")
print(f"Размерность эмбеддинга 2: {embedding2.shape}")

# Создание FAISS индекса
dimension = embedding1.shape[0]  # 384 для bge-small-en-v1.5
index = faiss.IndexFlatL2(dimension)  # Простой индекс с L2 расстоянием
faiss.normalize_L2(np.array([embedding1], dtype=np.float32))  # Нормализация
index.add(np.array([embedding1], dtype=np.float32))  # Добавляем первый эмбеддинг

# Поиск второго эмбеддинга
query_vector = np.array([embedding2], dtype=np.float32)
faiss.normalize_L2(query_vector)  # Нормализация запроса
distances, indices = index.search(query_vector, k=1)

# Вывод результата
distance = distances[0][0]
print(f"Сообщение 1: {message1}")
print(f"Сообщение 2: {message2}")
print(f"L2 расстояние между сообщениями: {distance:.4f}")
print(f"Рекомендуемый порог: попробуйте значения меньше {distance:.4f}")