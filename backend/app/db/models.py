from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    Text,
    Real,
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
    mastery_score = Column(Real, default=0.3)       # 原始掌握度 [0, 1]
    decay_rate = Column(Real, default=0.05)          # 衰减系数 λ
    total_attempts = Column(Integer, default=0)
    correct_count = Column(Integer, default=0)
    last_reviewed = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("user_id", "topic", name="uq_user_topic"),
    )


class KnowledgeConcept(Base):
    """知识库概念索引（Ingestion时从文档Heading提取）"""

    __tablename__ = "knowledge_concepts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    topic = Column(Text, nullable=False, unique=True)
    source_doc = Column(Text)
    heading_path = Column(Text)
    chunk_count = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)


class QuizSession(Base):
    """Quiz会话记录"""

    __tablename__ = "quiz_sessions"

    id = Column(Text, primary_key=True)
    user_id = Column(Text, nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime)
    total_questions = Column(Integer, default=0)
    correct_count = Column(Integer, default=0)
    topics_covered = Column(Text)                    # JSON数组
