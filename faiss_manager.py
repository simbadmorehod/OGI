from datetime import datetime, timedelta, timezone

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
        """Менеджер FAISS с IndexFlatL2 и IndexIDMap для точного поиска с ID и временными метками"""
        faiss.omp_set_num_threads(1)
        self.dimension = dimension + 1  # +1 для временной метки
        self.index_path = index_path
        self.clear_index()
        self.create_new_index()


    def create_new_index(self):
        """Создаёт новый FAISS индекс IndexFlatL2 с IndexIDMap"""
        logging.info("Создание нового FAISS индекса IndexFlatL2 с ID и временными метками...")
        base_index = faiss.IndexFlatL2(self.dimension)
        self.index = faiss.IndexIDMap(base_index)

    def load_index(self):
        """Загружает существующий FAISS индекс с диска"""
        logging.info(f"Загрузка FAISS индекса из {self.index_path}...")
        self.index = faiss.read_index(self.index_path)

    def add_vectors(self, ids: list, vectors: np.ndarray, timestamps: list):
        """Добавляет векторы и временные метки в FAISS с ID и прогресс-баром"""
        # Преобразуем временные метки в числовой формат
        timestamps_float = np.array([self.datetime_to_float(datetime.fromtimestamp(ts)) for ts in timestamps]).reshape(
            -1, 1)

        # Добавляем временные метки к векторам
        vectors_with_time = np.hstack((vectors, timestamps_float))
        vectors_with_time = np.ascontiguousarray(vectors_with_time.astype(np.float32))
        faiss.normalize_L2(vectors_with_time)

        logging.info(f"Добавление {len(ids)} векторов с временными метками в FAISS индекс...")
        with tqdm(total=len(ids), desc="Добавление векторов", unit="vec") as pbar:
            self.index.add_with_ids(vectors_with_time, np.array(ids, dtype=np.int64))
            pbar.update(len(ids))
        self.save_index()
        del vectors_with_time
        import gc
        gc.collect()

    def search_with_time_constraint(self, query_vector: np.ndarray, k: int = 5, time_constraint_days: int = 1):
        """Поиск в FAISS с учетом временного ограничения"""
        # Добавляем временную метку к запросу
        one_day_ago = datetime.now() - timedelta(days=time_constraint_days)
        query_with_time = np.hstack((query_vector, [self.datetime_to_float(one_day_ago)]))
        query_with_time = np.array(query_with_time, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_with_time)

        distances, indices = self.index.search(query_with_time, k)
        return [{"id": int(idx), "score": float(score)}
                for score, idx in zip(distances[0], indices[0]) if idx != -1]

    def datetime_to_float(self, dt):
        """Преобразует datetime в число секунд с 1970 года, учитывая часовой пояс"""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        total_seconds = (dt - epoch).total_seconds()
        return total_seconds

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