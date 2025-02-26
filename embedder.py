from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from torch.cuda.amp import autocast
import spacy


class StellaEmbedder:
    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(
            "dunzhang/stella_en_400M_v5",
            device=self.device,
            cache_folder="models/stella_en_400M_v5"
        )
        self.model.max_seq_length = 512  # Увеличиваем длину для Stella
        self.nlp = spacy.load("en_core_web_sm")  # Загружаем SpaCy для предобработки
        print(f"Embedding dim: {self.model.get_sentence_embedding_dimension()}")  # 768
        print(f"Модель загружена на устройство: {self.device}")

    def preprocess_text(self, text: str) -> str:
        """Предобработка текста через SpaCy: лемматизация и удаление стоп-слов"""
        doc = self.nlp(text)
        return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

    def embed(self, text: str):
        """Создание эмбеддинга для одного текста с предобработкой"""
        preprocessed_text = self.preprocess_text(text)
        if not preprocessed_text.strip():
            return np.zeros(768, dtype=np.float32)  # Размерность для Stella

        # Разбиение текста на части, если он длиннее max_seq_length
        sentences = [preprocessed_text[i:i + self.model.max_seq_length]
                     for i in range(0, len(preprocessed_text), self.model.max_seq_length)]
        with torch.no_grad(), autocast(enabled=(self.device == "cuda")):
            embeddings = self.model.encode(
                sentences,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            # Усреднение для длинных текстов
            return np.mean(embeddings, axis=0)

    def embed_batch(self, texts: list[str], batch_size: int = 1000):
        """Батчевая обработка текстов с GPU-оптимизацией"""
        if not texts or not all(isinstance(text, str) for text in texts):
            raise ValueError("❌ Ошибка: переданы некорректные данные (не все элементы - строки)")
        print(f"📥 Генерация эмбеддингов для {len(texts)} текстов...")

        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        embeddings = []

        for i in range(0, len(preprocessed_texts), batch_size):
            batch_texts = preprocessed_texts[i:i + batch_size]
            # Фильтруем пустые строки
            batch_texts = [t if t.strip() else " " for t in batch_texts]
            with torch.no_grad(), autocast(enabled=(self.device == "cuda")):
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_tensor=True,
                    normalize_embeddings=True
                )
            embeddings.append(batch_embeddings)
            if self.device == "cuda":
                torch.cuda.empty_cache()
                self.log_gpu_memory()

        return torch.cat(embeddings, dim=0)  # Объединяем батчи в один тензор

    def log_gpu_memory(self):
        """Логирование использования GPU"""
        if torch.cuda.is_available():
            print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
            print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB")


if __name__ == "__main__":
    embedder = StellaEmbedder(device="cuda")
    texts = ["ETH is growing fast!", "Ethereum is skyrocketing!"]
    embeddings = embedder.embed_batch(texts)
    print(f"Размер батча эмбеддингов: {embeddings.shape}")