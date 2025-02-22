import faiss
import numpy as np
import os

class FaissManager:
    def __init__(self, dimension: int = 384, nlist: int = 10, index_path="faiss_index.bin"):
        self.dimension = dimension
        self.nlist = nlist
        self.index_path = index_path  # Файл для хранения индекса

        self.quantizer = faiss.IndexFlatL2(dimension)
        self.index = None  # Инициализация без создания индекса

        # ✅ Пробуем загрузить индекс
        if not self.load_index():
            self.index = faiss.IndexIVFFlat(self.quantizer, dimension, nlist, faiss.METRIC_L2)
            print("⚠️ Новый FAISS индекс создан.")

    def train(self, data: np.ndarray):
        """Обучает FAISS и сохраняет индекс"""
        if not self.index.is_trained:
            print(f"🔄 Обучение FAISS индекса на {data.shape[0]} векторах...")
            self.index.train(data)
            self.save_index()  # ✅ Сохраняем после обучения

    def add_vectors(self, ids: np.ndarray, vectors: np.ndarray):
        """Добавляет векторы в индекс"""
        vectors = np.ascontiguousarray(vectors.astype('float32'))
        faiss.normalize_L2(vectors)

        if not self.index.is_trained:
            print("⚠️ Индекс не обучен, сначала обучаем...")
            self.train(vectors)

        self.index.add_with_ids(vectors, np.array(ids, dtype=np.int64))
        self.save_index()  # ✅ Сохраняем после добавления

    def search(self, query_vector: np.ndarray, k: int = 5):
        """Выполняет поиск в FAISS"""
        query_vector = query_vector.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, k)
        return [{"id": int(idx), "score": float(score)}
                for score, idx in zip(distances[0], indices[0]) if idx != -1]

    def save_index(self):
        """Сохраняет FAISS индекс в файл"""
        if self.index:
            faiss.write_index(self.index, self.index_path)
            print(f"✅ FAISS индекс сохранён в {self.index_path}")
        else:
            print("❌ Ошибка: индекс не существует, сохранить нечего.")

    def load_index(self):
        """Загружает FAISS индекс из файла, если он существует"""
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                print(f"📂 FAISS индекс загружен из {self.index_path}")
                return True
            except Exception as e:
                print(f"❌ Ошибка загрузки FAISS индекса: {e}")
                return False
        print("⚠️ Индекс не найден, создаётся новый.")
        return False

    def clear_index(self):
        """Удаляет старый индекс и создаёт новый"""
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        self.index = faiss.IndexIVFFlat(self.quantizer, self.dimension, self.nlist, faiss.METRIC_L2)
        print("🗑️ FAISS индекс сброшен и пересоздан.")