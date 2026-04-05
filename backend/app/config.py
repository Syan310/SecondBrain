from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # LLM
    llm_api_key: str = ""
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_model: str = "qwen3.6-plus"

    # Judge LLM (for evaluation)
    judge_api_key: str = ""
    judge_base_url: str = "https://api.moonshot.cn/v1"
    judge_model: str = "kimi-k2.5"

    # Embedding (bge-m3: 1024-dim, multilingual Chinese/English)
    embedding_model_path: str = "./models/bge-m3"

    # Reranker
    reranker_model_path: str = "./models/bge-reranker-v2-m3"

    # Milvus
    milvus_host: str = "milvus"
    milvus_port: int = 19530
    milvus_collection_name: str = "secondbrain_docs"
    milvus_memory_collection: str = "conversation_memory"

    # Memory
    memory_short_term_limit: int = 20       # max conversation rounds in short-term
    memory_consolidate_count: int = 10      # rounds to consolidate when limit reached
    memory_long_term_top_k: int = 3         # top-K long-term memories to retrieve
    memory_similarity_threshold: float = 0.3  # min similarity to include long-term memory

    # SQLite
    sqlite_db_path: str = "./data/secondbrain.db"

    # Notion
    notion_api_key: str = ""
    notion_database_id: str = ""

    # Mastery
    default_mastery_score: float = 0.3  # initial mastery for first-attempt topics

    # Upload
    max_upload_size_mb: int = 50  # max file upload size in MB

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_env: str = "development"
    secret_key: str = "change_this_to_a_random_secret_key"

    @property
    def database_url(self) -> str:
        return f"sqlite+aiosqlite:///{self.sqlite_db_path}"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


settings = Settings()
