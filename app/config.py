"""
Application configuration and settings.

Loads configuration from environment variables with sensible defaults.
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    # Extraction settings
    use_llm_extraction: bool = False
    llm_provider: str = "openai"  # openai or anthropic

    # API keys (optional - only needed if use_llm_extraction=True)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # Paths
    upload_dir: Path = Path("uploads")
    screenshot_dir: Path = Path("uploads/screenshots")

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
