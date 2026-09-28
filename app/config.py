"""
Application Configuration
Loads settings from environment variables and .env file.
"""

from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # API Keys
    GROQ_API_KEY: str = ""
    OPENAI_API_KEY: str = ""

    # Pipeline Providers
    STT_PROVIDER: Literal["groq", "openai", "mock"] = "groq"
    STT_MODEL: str = "whisper-large-v3-turbo"

    LLM_PROVIDER: Literal["groq", "openai", "mock"] = "groq"
    LLM_MODEL: str = "qwen/qwen3.8-27b"

    TTS_PROVIDER: Literal["edge-tts", "openai", "mock"] = "edge-tts"
    TTS_VOICE: str = "en-US-JennyNeural"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
