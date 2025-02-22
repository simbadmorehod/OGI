from database import get_db
from faiss_manager import FaissManager
import numpy as np
from models import MessageEmbeddings


def load_embeddings_to_faiss() -> FaissManager:
    faiss_manager = FaissManager(dimension=384)  # Новый размер

    # Загрузка данных батчами
    batch_size = 500
    offset = 0
    db = next(get_db())  # Получаем сессию базы данных

    while True:
        embeddings_data = db.query(MessageEmbeddings).offset(offset).limit(batch_size).all()
        if not embeddings_data:
            break

        ids = [emb.message_id for emb in embeddings_data]
        vectors = np.array([emb.embedding for emb in embeddings_data], dtype=np.float32)

        faiss_manager.add_vectors(ids, vectors)
        offset += batch_size

    print(f"✅ Всего загружено {faiss_manager.index.ntotal} векторов")
    return faiss_manager

if __name__ == "__main__":
    faiss_manager = load_embeddings_to_faiss()
    if faiss_manager.index.ntotal == 0:
        print("⚠️ FAISS загружен, но векторов нет. Возможно, база пустая.")
    else:
        print(f"✅ FAISS успешно загружен, {faiss_manager.index.ntotal} векторов.")