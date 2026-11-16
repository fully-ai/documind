"""Configuration for DocuMind."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCUMIND_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = Field(default=8092, ge=1, le=65535)
    data_dir: Path = Path("./data")
    ollama_base_url: str = "http://127.0.0.1:11434"
    embed_model: str = "nomic-embed-text"
    chat_model: str = "llama3.2"
    max_file_bytes: int = Field(default=20 * 1024 * 1024, ge=1024)
    chunk_size: int = Field(default=700, ge=100)
    chunk_overlap: int = Field(default=100, ge=0)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "documind.db"

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
