# Crypto Chat Assistant 🚀

Проект для анализа криптовалютных чатов с использованием современных NLP технологий

## 🛠 Технологии
- **Языковые модели**: DeepSeek-R1, BGE Embeddings
- **Поиск**: FAISS (Facebook AI Similarity Search)
- **Бэкенд**: FastAPI + Uvicorn
- **База данных**: PostgreSQL
- **Инфраструктура**: Python 3.9+, PyTorch, Transformers

## 📁 Структура проекта


## 🚀 Быстрый старт

### Установка
```bash
# 1. Клонировать репозиторий
git clone https://github.com/yourusername/crypto-chat-assistant.git
cd crypto-chat-assistant

# 2. Установить зависимости
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Запуск API сервера
uvicorn api.main:app --host 0.0.0.0 --port 800

# Дернуть за ручку API
curl -X POST "http://localhost:8000/api/v1/search" \
-H "Content-Type: application/json" \
-d '{"query": "BTC price prediction", "top_k":5,
  "detailed": true}'

# Алгоритм обработки запроса
graph TD
    A[Пользователь] --> B[FastAPI]
    B --> C{Поисковый движок}
    C --> D[FAISS Index]
    C --> E[PostgreSQL]
    B --> F[DeepSeek LLM]
    D --> G[BGE Embeddings]
    F --> H[Анализ контекста]
    H --> I[Формирование ответа]
