import numpy as np
from sentence_transformers import SentenceTransformer
import torch


class BGEEmbedder:
    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(
            "BAAI/bge-small-en-v1.5",  # ✅ Меньшая размерность
            device=self.device,
            cache_folder="models/bge-small-en"
        )
        self.model.max_seq_length = 256  # Оптимизация длины текста
        print(f"Embedding dim: {self.model.get_sentence_embedding_dimension()}")  # 384

    def embed(self, text: str):
        if not text.strip():
            return np.zeros(384, dtype=np.float32)  # Возвращаем нулевой вектор

        with torch.no_grad():
            return self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                precision="fp16"  # Экономия памяти
            )

    def embed_batch(self, texts: list[str]):
        if not texts or not all(isinstance(text, str) for text in texts):
            raise ValueError("❌ Ошибка: переданы некорректные данные (не все элементы - строки)")

        print(f"📥 Генерация эмбеддингов для {len(texts)} текстов...")

        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()