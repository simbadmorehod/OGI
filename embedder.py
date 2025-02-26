from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import spacy
import torch
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
        """Инициализация с указанием модели SpaCy"""
        self.nlp_model_name = model_name  # Имя модели SpaCy
        self.device = device


    def preprocess_text(self, text: str) -> str:
        """Предобработка текста: лемматизация и удаление стоп-слов"""
        nlp = spacy.load(self.nlp_model_name)  # Загружаем модель в каждом потоке
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

    def embed(self, text: str):
        """Создание эмбеддинга для одного текста с предобработкой"""
        logging.info("Генерация эмбеддинга для одного текста...")
        preprocessed_text = self.preprocess_text(text)
        if not preprocessed_text.strip():
            logging.warning("Предобработанный текст пуст. Возвращается нулевой вектор.")
            return np.zeros(1024, dtype=np.float32)

        sentences = [preprocessed_text[i:i + self.model.max_seq_length]
                     for i in range(0, len(preprocessed_text), self.model.max_seq_length)]
        with torch.no_grad(), autocast('cuda', enabled=(self.device)):
            embeddings = self.model.encode(
                sentences,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            result = np.mean(embeddings, axis=0)
            logging.info("Эмбеддинг успешно сгенерирован.")
            return result

    def embed_batch(self, texts: list[str], max_workers: int = 8) -> list[str]:
        """Параллельная предобработка текстов с ThreadPoolExecutor"""
        logging.info(f"Обработка {len(texts)} текстов на {max_workers} потоках...")

        # Используем partial для передачи функции предобработки
        preprocess_func = partial(self.preprocess_text)

        # Параллельная обработка с 8 потоками
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            preprocessed_texts = list(tqdm(
                executor.map(preprocess_func, texts),
                total=len(texts),
                desc="Предобработка текстов"
            ))

        logging.info("Предобработка завершена.")
        return preprocessed_texts

    def log_gpu_memory(self):
        """Логирование использования GPU"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e6
            reserved = torch.cuda.memory_reserved() / 1e6
            logging.info(f"GPU Memory Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")


# Пример использования
if __name__ == "__main__":
    # Создаем экземпляр процессора с моделью для русского языка
    processor = StellaEmbedder(model_name="ru_core_news_sm")

    # Пример текстов (дублируем для объема)
    texts = ["Привет, мир!", "Это тест.", "Параллельная обработка крута.", "Еще один текст."] * 250

    # Запуск обработки на 8 ядрах
    result = processor.embed_batch(texts, max_workers=8)
    print(f"Обработано {len(result)} текстов.")