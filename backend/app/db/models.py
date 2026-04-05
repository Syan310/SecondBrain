from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Float,
    Integer,
    Text,
    DateTime,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class UserMastery(Base):
    """知识点掌握度（带时间衰减）"""

    __tablename__ = "user_mastery"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Text, nullable=False)
    topic = Column(Text, nullable=False)
    mastery_score = Column(Float, default=0.3)       # 原始掌握度 [0, 1]
    decay_rate = Column(Float, default=0.05)          # 衰减系数 λ
    total_attempts = Column(Integer, default=0)
    correct_count = Column(Integer, default=0)
    last_reviewed = Column(DateTime)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint("user_id", "topic", name="uq_user_topic"),
    )


class KnowledgeConcept(Base):
    """知识库概念索引（Ingestion时从文档Heading提取）"""

    __tablename__ = "knowledge_concepts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    topic = Column(Text, nullable=False)
    source_doc = Column(Text, nullable=False)
    heading_path = Column(Text)
    chunk_count = Column(Integer, default=1)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint("topic", "source_doc", name="uq_topic_source"),
    )


class ConversationHistory(Base):
    """短期对话记录（最近 N 轮，FIFO 淘汰）"""

    __tablename__ = "conversation_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Text, nullable=False)
    session_id = Column(Text, nullable=False)
    role = Column(Text, nullable=False)           # "user" | "assistant"
    content = Column(Text, nullable=False)
    round_number = Column(Integer, nullable=False)  # 对话轮次，同一轮 user+assistant 共享
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class QuizSession(Base):
    """Quiz会话记录"""

    __tablename__ = "quiz_sessions"

    id = Column(Text, primary_key=True)
    user_id = Column(Text, nullable=False)
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    ended_at = Column(DateTime)
    total_questions = Column(Integer, default=0)
    correct_count = Column(Integer, default=0)
    topics_covered = Column(Text)                    # JSON数组
