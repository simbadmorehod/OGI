import numpy as np
import faiss
from embedder import StellaEmbedder

# Ожидаемая размерность для Stella_en_400M_v5
EXPECTED_DIMENSION = 768

# Инициализация эмбеддера
embedder = StellaEmbedder(device="cpu")  # Используйте "cuda", если есть GPU

# Тестовые сообщения
message1 = "ETH растет как на стероидах, скоро будет $5000!"
message2 = "Ethereum взлетает, цена может дойти до 5000 долларов."

# Генерация эмбеддингов
embedding1 = embedder.embed(message1)
embedding2 = embedder.embed(message2)

# Проверка размерности
print(f"Размерность эмбеддинга 1: {embedding1.shape}")
print(f"Размерность эмбеддинга 2: {embedding2.shape}")

if embedding1.shape[0] != EXPECTED_DIMENSION or embedding2.shape[0] != EXPECTED_DIMENSION:
    raise ValueError(
        f"Ошибка: размерность эмбеддингов ({embedding1.shape[0]}) не соответствует ожидаемой ({EXPECTED_DIMENSION})"
    )

# Создание FAISS индекса
dimension = embedding1.shape[0]  # Должно быть 768 для Stella_en_400M_v5
index = faiss.IndexFlatL2(dimension)  # Простой индекс с L2 расстоянием

# Подготовка первого эмбеддинга
embedding1_array = np.array([embedding1], dtype=np.float32)
faiss.normalize_L2(embedding1_array)  # Нормализация
index.add(embedding1_array)  # Добавляем первый эмбеддинг

# Подготовка второго эмбеддинга для поиска
query_vector = np.array([embedding2], dtype=np.float32)
faiss.normalize_L2(query_vector)  # Нормализация запроса

# Поиск
distances, indices = index.search(query_vector, k=1)

# Вывод результата
distance = distances[0][0]
print(f"Сообщение 1: {message1}")
print(f"Сообщение 2: {message2}")
print(f"L2 расстояние между сообщениями: {distance:.4f}")
print(f"Рекомендуемый порог: попробуйте значения меньше {distance:.4f}")