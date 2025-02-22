from sqlalchemy import Column, Integer, BigInteger, String, DateTime, Float, Boolean, ARRAY, ForeignKey, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Messages(Base):
    __tablename__ = "messages"
    message_id = Column(BigInteger, primary_key=True, unique=True)
    text_message = Column(String(8888), nullable=True)
    tg_user_id_sender = Column(BigInteger, nullable=False)
    tg_user_id_receiver = Column(BigInteger, nullable=False)
    phone_user_sender = Column(String(88), nullable=True)
    sender_bio = Column(String(250), nullable=True)
    username_sender = Column(String(88), nullable=True)
    date_message = Column(DateTime(timezone=True))
    date_creation = Column(DateTime(timezone=True), nullable=True)
    local_date_creation = Column(DateTime(timezone=True), server_default=func.now())
    source_id = Column(String, nullable=False)
    source_title = Column(String(250), nullable=True)
    message_url = Column(String(250), nullable=True)
    full_name_sender = Column(String(88), nullable=True)
    target_keys = Column(String(20000), nullable=True)
    company_id = Column(BigInteger, nullable=False)
    company_title = Column(String(888), nullable=True)
    read = Column(Boolean, default=True)
    unique_hash = Column(String(888), unique=True, nullable=True)
    toxicity = Column(Float, nullable=True)  # Новое поле для уровня токсичности
    embedding = relationship("MessageEmbeddings", back_populates="message")

class MessageEmbeddings(Base):
    __tablename__ = "message_embeddings"
    id = Column(Integer, primary_key=True)
    message_id = Column(BigInteger, ForeignKey('messages.message_id'), unique=True)  # Изменено на BigInteger
    embedding = Column(ARRAY(Float), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    message = relationship("Messages", back_populates="embedding")