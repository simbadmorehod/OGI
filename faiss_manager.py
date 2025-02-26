import faiss
import numpy as np
import os

class FaissManager:
    def __init__(self, dimension: int = 768, nlist: int = 10, index_path="faiss_index.bin"):
        """Менеджер FAISS с созданием нового индекса при запуске"""
        faiss.omp_set_num_threads(1)  # Ограничиваем потоки для стабильности
        self.dimension = dimension
        self.nlist = nlist
        self.index_path = index_path  # Файл для хранения индекса

        # ❌ Каждый запуск - создаем новый индекс
        self.clear_index()
        self.create_new_index()

    def create_new_index(self):
        """Создаёт новый FAISS индекс"""
        print("⚠️ Новый FAISS индекс создаётся.")
        quantizer = faiss.IndexFlatL2(self.dimension)  # Квантайзер
        self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, faiss.METRIC_L2)

    def train(self, data: np.ndarray):
        """Обучает FAISS индекс"""
        if not self.index.is_trained:
            print(f"🔄 Обучение FAISS индекса на {data.shape[0]} векторах...")
            self.index.train(data)
            self.save_index()

    def add_vectors(self, ids: list, vectors: np.ndarray):
        """Добавляет векторы в FAISS"""
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        faiss.normalize_L2(vectors)

        if not self.index.is_trained:
            self.train(vectors)

        self.index.add_with_ids(vectors, np.array(ids, dtype=np.int64))
        self.save_index()

    def search(self, query_vector: np.ndarray, k: int = 5):
        """Поиск в FAISS"""
        query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, k)
        return [{"id": int(idx), "score": float(score)}
                for score, idx in zip(distances[0], indices[0]) if idx != -1]

    def save_index(self):
        """Сохраняет FAISS индекс"""
        if self.index and self.index.is_trained:
            faiss.write_index(self.index, self.index_path)
            print(f"✅ FAISS индекс сохранён: {self.index_path}")
        else:
            print("❌ Ошибка: индекс не обучен или не существует.")

    def clear_index(self):
        """Удаляет старый индекс и создаёт новый"""
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        print("🗑️ FAISS индекс сброшен и пересоздан.")