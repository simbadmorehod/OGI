import numpy as np
import spacy
import torch
from sentence_transformers import SentenceTransformer
from torch.cuda.amp import autocast
from tqdm import tqdm
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class StellaEmbedder:
    def __init__(self, model_name="ru_core_news_sm", device=torch.device("cuda")):
        """Инициализация с указанием модели SpaCy и устройства"""
        self.nlp_model_name = model_name  # Имя модели SpaCy (если предобработка нужна)
        self.device = device
        logging.info(f"Выбрано устройство: {self.device}")
        # Загрузка модели SentenceTransformer на GPU
        self.model = SentenceTransformer("intfloat/multilingual-e5-large-instruct", device=self.device)
        logging.info("Модель SentenceTransformer загружена.")

    def preprocess_text(self, text: str) -> str:
        """Предобработка текста: лемматизация и удаление стоп-слов (опционально)"""
        logging.info("Предобработка текста на CPU (опционально)...")
        nlp = spacy.load(self.nlp_model_name)  # Загружаем модель SpaCy
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

    def embed(self, text: str) -> np.ndarray:
        """Создание эмбеддинга для одного текста"""
        logging.info("Генерация эмбеддинга для одного текста...")
        # Предобработка опциональна, можно её убрать для полной работы на GPU
        # preprocessed_text = self.preprocess_text(text)
        preprocessed_text = text  # Пропускаем предобработку для скорости
        if not preprocessed_text.strip():
            logging.warning("Текст пуст. Возвращается нулевой вектор.")
            return np.zeros(1024, dtype=np.float32)  # Обновлено с 1024 на 768

        with torch.no_grad(), autocast(enabled=True):
            embedding = self.model.encode(
                preprocessed_text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            logging.info("Эмбеддинг успешно сгенерирован.")
            return embedding

    def embed_batch(self, texts: list[str], max_workers: int = 8) -> np.ndarray:
        """Генерация эмбеддингов для батча текстов на GPU"""
        logging.info(f"Генерация эмбеддингов для {len(texts)} текстов на GPU...")
        # Предобработка опциональна, здесь мы её пропускаем для скорости
        # preprocess_func = partial(self.preprocess_text)
        # with ThreadPoolExecutor(max_workers=max_workers) as executor:
        #     preprocessed_texts = list(tqdm(
        #         executor.map(preprocess_func, texts),
        #         total=len(texts),
        #         desc="Предобработка текстов"
        #     ))
        preprocessed_texts = texts  # Пропускаем предобработку

        with torch.no_grad(), autocast(enabled=True):
            embeddings = self.model.encode(
                preprocessed_texts,
                batch_size=20000,  # Батчевая обработка для ускорения
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        logging.info("Все эмбеддинги сгенерированы.")
        return embeddings

    def log_gpu_memory(self):
        """Логирование использования GPU"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e6
            reserved = torch.cuda.memory_reserved() / 1e6
            logging.info(f"GPU Memory Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

# Пример использования
if __name__ == "__main__":
    embedder = StellaEmbedder(model_name="ru_core_news_sm", device=torch.device("cuda"))
    texts = ["Привет, мир!", "Это тест.", "Параллельная обработка крута.", "Еще один текст."] * 250
    embeddings = embedder.embed_batch(texts, max_workers=8)
    print(f"Обработано {embeddings.shape[0]} текстов.")
    embedder.log_gpu_memory()