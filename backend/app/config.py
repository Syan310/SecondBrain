from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # LLM
    llm_api_key: str = ""
    llm_base_url: str = "https://dashscope-us.aliyuncs.com/compatible-mode/v1"
    llm_model: str = "qwen3.6-plus"

    # Judge LLM (for evaluation)
    judge_api_key: str = ""
    judge_base_url: str = "https://api.moonshot.cn/v1"
    judge_model: str = "kimi-k2.5"

    # OpenRouter provider routing (provider slugs, fallbacks disabled; empty = no routing)
    llm_provider_order: list[str] = ["deepseek"]
    judge_provider_order: list[str] = ["anthropic"]

    # Embedding (bge-m3: 1024-dim, multilingual Chinese/English)
    embedding_model_path: str = "./models/bge-m3"

    # Reranker
    reranker_model_path: str = "./models/bge-reranker-v2-m3"
    enable_reranker: bool = False
    reranker_use_heading: bool = False  # prepend heading_path to reranker passages

    # Citations: "off" | "prompt" (strip markers) | "filter" (drop uncited sentences)
    #   | "fallback" (keep sentences; if the notes have no answer, say so, then answer from model knowledge)
    citation_mode: str = "fallback"

    # Milvus
    milvus_host: str = "milvus"
    milvus_port: int = 19530
    milvus_collection_name: str = "secondbrain_docs"
    milvus_memory_collection: str = "conversation_memory"

    # BM25 (set when the collection is created; changing them needs a re-ingest)
    bm25_k1: float = 1.2   # term-frequency saturation
    bm25_b: float = 0.75   # document-length normalization

    # Memory
    memory_short_term_limit: int = 20       # max conversation rounds in short-term
    memory_consolidate_count: int = 10      # rounds to consolidate when limit reached
    memory_long_term_top_k: int = 3         # top-K long-term memories to retrieve
    memory_similarity_threshold: float = 0.3  # min similarity to include long-term memory

    # SQLite
    sqlite_db_path: str = "./data/secondbrain.db"

    # Mastery
    default_mastery_score: float = 0.3  # initial mastery for first-attempt topics

    # Upload
    max_upload_size_mb: int = 50  # max file upload size in MB

    # Langfuse (optional — tracing disabled when keys are empty)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

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
        # Allow older/unused keys in .env without crashing startup.
        # This is important for local dev where env files may accumulate settings.
        "extra": "ignore",
    }


settings = Settings()
