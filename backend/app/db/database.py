import logging

from sqlalchemy import inspect as sa_inspect, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

logger = logging.getLogger(__name__)

engine = create_async_engine(
    settings.database_url,
    echo=settings.app_env == "development",
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


def _needs_concept_migration(connection) -> bool:
    """Check if knowledge_concepts table has old schema (source_doc column)."""
    inspector = sa_inspect(connection)
    if not inspector.has_table("knowledge_concepts"):
        return False
    columns = {col["name"] for col in inspector.get_columns("knowledge_concepts")}
    return "source_doc" in columns


def _migrate_concepts(connection):
    """Drop old concept tables and let create_all rebuild with new schema."""
    if _needs_concept_migration(connection):
        logger.warning("Detected old knowledge_concepts schema — migrating (drop + recreate).")
        connection.execute(text("DROP TABLE IF EXISTS concept_sources"))
        connection.execute(text("DROP TABLE IF EXISTS knowledge_concepts"))


async def init_db():
    """创建所有数据库表（含 schema 迁移）"""
    from app.db.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(_migrate_concepts)
        await conn.run_sync(Base.metadata.create_all)
