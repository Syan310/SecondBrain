import logging
from pathlib import Path

from sqlalchemy import event, inspect as sa_inspect, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

logger = logging.getLogger(__name__)


_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent  # backend/


def _sqlite_async_url() -> str:
    """Resolve SQLite path, ensure parent dir exists (gitignored ./data/ is often missing)."""
    p = Path(settings.sqlite_db_path).expanduser()
    if not p.is_absolute():
        p = (_BACKEND_DIR / p).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite+aiosqlite:///{p.as_posix()}"


engine = create_async_engine(
    _sqlite_async_url(),
    echo=settings.app_env == "development",
)


@event.listens_for(engine.sync_engine, "connect")
def _set_sqlite_pragmas(dbapi_conn, connection_record):
    """Enable WAL journal mode and busy timeout for concurrent access."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


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
