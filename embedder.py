from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from torch.cuda.amp import autocast
import spacy
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class StellaEmbedder:
    def __init__(self, device=None):
        logging.info("Инициализация StellaEmbedder...")
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Выбрано устройство: {self.device}")

        logging.info("Загрузка модели SentenceTransformer...")
        self.model = SentenceTransformer(
            "dunzhang/stella_en_400M_v5",
            device=self.device,
            cache_folder="models/stella_en_400M_v5",
            trust_remote_code=True
        )
        self.model.max_seq_length = 512
        logging.info(f"Модель загружена на устройство: {self.device}")
        logging.info(f"Размерность эмбеддингов: {self.model.get_sentence_embedding_dimension()}")

        logging.info("Загрузка модели SpaCy...")
        self.nlp = spacy.load("en_core_web_sm")
        logging.info("Модель SpaCy загружена.")

    def preprocess_text(self, text: str) -> str:
        """Предобработка текста через SpaCy: лемматизация и удаление стоп-слов"""
        logging.info("Начало предобработки текста...")
        doc = self.nlp(text)
        result = " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
        logging.info("Предобработка текста завершена.")
        return result

    def embed(self, text: str):
        """Создание эмбеддинга для одного текста с предобработкой"""
        logging.info("Генерация эмбеддинга для одного текста...")
        preprocessed_text = self.preprocess_text(text)
        if not preprocessed_text.strip():
            logging.warning("Предобработанный текст пуст. Возвращается нулевой вектор.")
            return np.zeros(1024, dtype=np.float32)

        sentences = [preprocessed_text[i:i + self.model.max_seq_length]
                     for i in range(0, len(preprocessed_text), self.model.max_seq_length)]
        with torch.no_grad(), autocast('cuda', enabled=(self.device == "cuda")):
            embeddings = self.model.encode(
                sentences,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            result = np.mean(embeddings, axis=0)
            logging.info("Эмбеддинг успешно сгенерирован.")
            return result

    def embed_batch(self, texts: list[str], batch_size: int = 1000):
        """Батчевая обработка текстов с GPU-оптимизацией"""
        if not texts or not all(isinstance(text, str) for text in texts):
            raise ValueError("❌ Ошибка: переданы некорректные данные (не все элементы - строки)")
        logging.info(f"Генерация эмбеддингов для {len(texts)} текстов...")

        logging.info("Начало предобработки текстов...")
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        logging.info("Предобработка текстов завершена.")
        embeddings = []

        for i in range(0, len(preprocessed_texts), batch_size):
            logging.info(f"Обработка батча {i//batch_size + 1} из {len(preprocessed_texts)//batch_size + 1}...")
            batch_texts = preprocessed_texts[i:i + batch_size]
            batch_texts = [t if t.strip() else " " for t in batch_texts]
            with torch.no_grad(), autocast('cuda', enabled=(self.device == "cuda")):
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_tensor=True,
                    normalize_embeddings=True
                )
            embeddings.append(batch_embeddings)
            if self.device == "cuda":
                torch.cuda.empty_cache()
                self.log_gpu_memory()
            logging.info(f"Батч {i//batch_size + 1} обработан.")

        result = torch.cat(embeddings, dim=0)
        logging.info("Все эмбеддинги сгенерированы.")
        return result

    def log_gpu_memory(self):
        """Логирование использования GPU"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e6
            reserved = torch.cuda.memory_reserved() / 1e6
            logging.info(f"GPU Memory Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

if __name__ == "__main__":
    embedder = StellaEmbedder(device="cuda")
    texts = ["ETH is growing fast!", "Ethereum is skyrocketing!"]
    embeddings = embedder.embed_batch(texts)
    logging.info(f"Размер батча эмбеддингов: {embeddings.shape}")