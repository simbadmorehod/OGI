import faiss
import numpy as np
import os
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

class FaissManager:
    def __init__(self, dimension: int = 1024, index_path="faiss_index.bin"):
        """Менеджер FAISS с IndexFlatL2 и IndexIDMap для точного поиска с ID"""
        faiss.omp_set_num_threads(1)
        self.dimension = dimension
        self.index_path = index_path
        self.clear_index()
        self.create_new_index()

    def create_new_index(self):
        """Создаёт новый FAISS индекс IndexFlatL2 с IndexIDMap"""
        logging.info("Создание нового FAISS индекса IndexFlatL2 с ID...")
        base_index = faiss.IndexFlatL2(self.dimension)  # Базовый точный индекс
        self.index = faiss.IndexIDMap(base_index)  # Обёртка для поддержки ID

    def load_index(self):
        """Загружает существующий FAISS индекс с диска"""
        logging.info(f"Загрузка FAISS индекса из {self.index_path}...")
        self.index = faiss.read_index(self.index_path)

    def add_vectors(self, ids: list, vectors: np.ndarray):
        """Добавляет векторы в FAISS с ID и прогресс-баром"""
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        faiss.normalize_L2(vectors)

        logging.info(f"Добавление {len(ids)} векторов в FAISS индекс...")
        with tqdm(total=len(ids), desc="Добавление векторов", unit="vec") as pbar:
            self.index.add_with_ids(vectors, np.array(ids, dtype=np.int64))
            pbar.update(len(ids))
        self.save_index()
        del vectors
        import gc
        gc.collect()

    def search(self, query_vector: np.ndarray, k: int = 5):
        """Поиск в FAISS"""
        query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, k)
        return [{"id": int(idx), "score": float(score)}
                for score, idx in zip(distances[0], indices[0]) if idx != -1]

    def save_index(self):
        """Сохраняет FAISS индекс"""
        if self.index:
            faiss.write_index(self.index, self.index_path)
            logging.info(f"FAISS индекс сохранён в {self.index_path}")
        else:
            logging.warning("Индекс не существует, сохранение пропущено.")

    def clear_index(self):
        """Удаляет старый индекс и создаёт новый"""
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
            logging.info(f"Старый FAISS индекс удалён: {self.index_path}")
        else:
            logging.info("Файл FAISS индекса не найден, очистка не требуется.")
        self.create_new_index()