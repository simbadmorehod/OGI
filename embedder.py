import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from torch.cuda.amp import autocast

class BGEEmbedder:
    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(
            "BAAI/bge-small-en-v1.5",
            device=self.device,
            cache_folder="models/bge-small-en"
        )
        self.model.max_seq_length = 256
        print(f"Embedding dim: {self.model.get_sentence_embedding_dimension()}")  # 384
        print(f"Модель загружена на устройство: {self.device}")

    def embed(self, text: str):
        if not text.strip():
            return np.zeros(384, dtype=np.float32)
        with torch.no_grad(), autocast(enabled=(self.device == "cuda")):
            return self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

    def embed_batch(self, texts: list[str], batch_size: int = 256):
        if not texts or not all(isinstance(text, str) for text in texts):
            raise ValueError("❌ Ошибка: переданы некорректные данные (не все элементы - строки)")
        print(f"📥 Генерация эмбеддингов для {len(texts)} текстов...")
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            with torch.no_grad(), autocast(enabled=(self.device == "cuda")):
                batch_embeddings = self.model.encode(batch_texts, convert_to_numpy=True, normalize_embeddings=True)
            embeddings.extend(batch_embeddings)
            if self.device == "cuda":
                torch.cuda.empty_cache()
                self.log_gpu_memory()
        return embeddings

    def log_gpu_memory(self):
        if torch.cuda.is_available():
            print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
            print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB")