from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from torch.cuda.amp import autocast
import spacy
from langdetect import detect, DetectorFactory

# Для стабильности результатов langdetect
DetectorFactory.seed = 0


class StellaEmbedder:
    def __init__(self, device=None):
        # Определяем устройство (GPU или CPU)
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        # Загружаем модель SentenceTransformer
        self.model = SentenceTransformer(
            "dunzhang/stella_en_400M_v5",
            device=self.device,
            cache_folder="models/stella_en_400M_v5",
            trust_remote_code=True
        )
        self.model.max_seq_length = 512
        # Загружаем модели SpaCy для обоих языков
        self.nlp_en = spacy.load("en_core_web_sm")
        self.nlp_ru = spacy.load("ru_core_news_sm")
        print(f"Размерность эмбеддингов: {self.model.get_sentence_embedding_dimension()}")
        print(f"Модель загружена на устройство: {self.device}")

    def detect_language(self, text: str) -> str:
        """Определяет язык текста: 'en' или 'ru'"""
        try:
            lang = detect(text)
            if lang in ['en', 'ru']:
                return lang
            return 'en'  # По умолчанию английский, если язык не распознан как 'ru'
        except:
            return 'en'  # В случае ошибки считаем текст английским

    def preprocess_text(self, text: str) -> str:
        """Предобработка текста с учётом языка"""
        lang = self.detect_language(text)
        # Выбираем модель SpaCy в зависимости от языка
        nlp = self.nlp_en if lang == 'en' else self.nlp_ru
        doc = nlp(text)
        # Лемматизация, удаление стоп-слов и пунктуации
        return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

    def embed(self, text: str):
        """Генерация эмбеддинга для одного текста"""
        preprocessed_text = self.preprocess_text(text)
        if not preprocessed_text.strip():
            return np.zeros(768, dtype=np.float32)

        # Разбиваем текст на части, если он длиннее max_seq_length
        sentences = [preprocessed_text[i:i + self.model.max_seq_length]
                     for i in range(0, len(preprocessed_text), self.model.max_seq_length)]
        with torch.no_grad(), autocast(enabled=(self.device == "cuda")):
            embeddings = self.model.encode(
                sentences,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return np.mean(embeddings, axis=0)  # Усредняем, если несколько частей

    def embed_batch(self, texts: list[str], batch_size: int = 1000):
        """Генерация эмбеддингов для батча текстов"""
        if not texts or not all(isinstance(text, str) for text in texts):
            raise ValueError("❌ Ошибка: переданы некорректные данные (не все элементы - строки)")
        print(f"📥 Генерация эмбеддингов для {len(texts)} текстов...")

        # Предобработка всех текстов с учётом их языка
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        embeddings = []

        # Обрабатываем батчи
        for i in range(0, len(preprocessed_texts), batch_size):
            batch_texts = preprocessed_texts[i:i + batch_size]
            batch_texts = [t if t.strip() else " " for t in batch_texts]  # Заменяем пустые строки
            with torch.no_grad(), autocast(enabled=(self.device == "cuda")):
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_tensor=True,
                    normalize_embeddings=True
                )
            embeddings.append(batch_embeddings)
            if self.device == "cuda":
                torch.cuda.empty_cache()  # Очищаем память GPU

        return torch.cat(embeddings, dim=0)


if __name__ == "__main__":
    embedder = StellaEmbedder(device="cuda")
    texts = ["ETH is growing fast!", "Ethereum взлетает!"]
    embeddings = embedder.embed_batch(texts)
    print(f"Размер батча эмбеддингов: {embeddings.shape}")