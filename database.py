from sqlalchemy import select, func, and_, create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
from models import Messages, Base

# Замените DATABASE_URL на ваши реальные данные
DATABASE_URL = "postgresql://grepdrop:asdy7idf7v6gf@localhost:5432/grepdrop"

# Настраиваем engine с увеличенным пулом соединений
engine = create_engine(
    DATABASE_URL,
    pool_size=10,  # начальное количество соединений
    max_overflow=20,  # дополнительные соединения сверх pool_size
    pool_timeout=30
)

# Создаем таблицы один раз при инициализации
Base.metadata.create_all(bind=engine)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_filtered_messages(db, keywords: list[str], time_filter: str):
    """Получение сообщений с фильтрацией по ключевым словам и времени."""
    query = select(Messages)
    if keywords:
        # Первое ключевое слово
        query = query.where(Messages.text_message.ilike(f"%{keywords[0]}%"))
        # Остальные ключевые слова через OR
        for keyword in keywords[1:]:
            query = query.or_(Messages.text_message.ilike(f"%{keyword}%"))
    if time_filter == "last_week":
        start_date = datetime.now() - timedelta(days=7)
        query = query.where(Messages.date_message >= start_date)
    elif time_filter == "last_month":
        start_date = datetime.now() - timedelta(days=30)
        query = query.where(Messages.date_message >= start_date)

    result = db.execute(query)
    return result.scalars().all()