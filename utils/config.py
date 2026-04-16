"""Configuration management for HireFlow project."""

from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class AppConfig(BaseSettings):
    """Minimal application configuration (only what's used)."""

    PINECONE_API_KEY: str = Field(default="", env="PINECONE_API_KEY")
    PINECONE_INDEX_NAME: str = Field(default="hireflow", env="PINECONE_INDEX_NAME")
    PINECONE_DIMENSION: int = Field(default=384, env="PINECONE_DIMENSION")
    PINECONE_METRIC: str = Field(default="cosine", env="PINECONE_METRIC")

    GOOGLE_API_KEY: str = Field(default="", env="GOOGLE_API_KEY")
    LLM_MODEL: str = Field(default="gemini-2.5-flash-lite", env="GOOGLE_MODEL")

    MAX_TEXT_LENGTH: int = Field(default=4000, env="MAX_TEXT_LENGTH")

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

config = AppConfig()

GOOGLE_API_KEY = config.GOOGLE_API_KEY
LLM_MODEL = config.LLM_MODEL

PINECONE_API_KEY = config.PINECONE_API_KEY
PINECONE_INDEX_NAME = config.PINECONE_INDEX_NAME
PINECONE_DIMENSION = config.PINECONE_DIMENSION
PINECONE_METRIC = config.PINECONE_METRIC

MAX_TEXT_LENGTH = config.MAX_TEXT_LENGTH
