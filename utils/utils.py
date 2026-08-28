"""
Common utility functions for text processing, PDF handling, and error detection.
Centralized utilities used across the HireFlow project.
"""

import logging
import re
from langchain_community.document_loaders import PyPDFLoader

# Third-party loggers that emit a line per HTTP request. They are chatty enough
# to bury HireFlow's own output — sentence-transformers alone logs a dozen
# huggingface.co requests every time the embedding model loads.
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "huggingface_hub",
    "sentence_transformers",
    "transformers",
    "filelock",
    "pinecone",
    "openai",
)

_logging_configured = False


def _configure_logging() -> None:
    """Set up the root handler once: WARNING for libraries, INFO for HireFlow.

    basicConfig sets the level on the *root* logger, so calling it with
    level=INFO (as this used to) turns on INFO for every third-party library
    too. The root handler is left at WARNING instead, and HireFlow's own
    loggers opt in to INFO individually — a record still reaches the root
    handler regardless of the root logger's level, because propagation only
    re-checks handler levels, not ancestor logger levels.
    """
    global _logging_configured
    if _logging_configured:
        return

    logging.basicConfig(
        level=logging.WARNING,
        format='[%(asctime)s] %(levelname)s: %(message)s',
    )
    for noisy in _NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """Create standardized logger with timestamp formatting"""
    _configure_logging()
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger

def clean_text(text: str) -> str:
    """Normalize text by removing extra whitespace and invalid characters"""
    if not text:
        return ""
    
    # Normalize whitespace (multiple spaces/tabs/newlines to single space)
    text = re.sub(r'\s+', ' ', text)
    # Keep only alphanumeric, common punctuation, and symbols
    text = re.sub(r'[^\w\s\.\,\-\+\@\#\&\*\(\)]', '', text)
    return text.strip()

def truncate_text(text: str, max_length: int = 8000) -> str:
    """Cut text at max length and add ellipsis if truncated"""
    if not text:
        return ""
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length] + "..."

def load_pdf(file_path: str) -> str:
    """Extract all text content from PDF file using LangChain"""
    logger = get_logger(__name__)
    
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        if not documents:
            return ""
        
        # Combine all pages into single text
        full_text = "\n".join([doc.page_content for doc in documents])
        return full_text.strip()
        
    except Exception as e:
        logger.error(f"Error loading PDF {file_path}: {e}")
        return ""

def is_quota_error(error: Exception) -> bool:
    """Detect API quota/rate limit errors for graceful handling"""
    error_str = str(error).lower()
    return "429" in str(error) or "quota" in error_str or "rate limit" in error_str
