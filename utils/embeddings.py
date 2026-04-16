"""Google Generative AI embedding model initialization.

Handles API quota errors and ensures an asyncio event loop exists when
called from threads (Streamlit runs app code in a worker thread).
"""

import asyncio
import logging
from typing import Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.utils import is_quota_error

logger = logging.getLogger(__name__)


def _ensure_event_loop() -> None:
    """Ensure there's an asyncio event loop for the current thread.

    Streamlit runs user code in a worker thread (`ScriptRunner.scriptThread`).
    Some LLM/embedding clients rely on asyncio and call `asyncio.get_running_loop()`
    (or similar). If no loop is present, Python raises RuntimeError. Create and
    set a new loop for the current thread to avoid that error.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


def get_embeddings() -> Optional[HuggingFaceEmbeddings]:
    """Create Google embedding model with quota error handling.

    Returns `None` if embeddings cannot be created (quota or other errors).
    """
    try:
        _ensure_event_loop()

        # return GoogleGenerativeAIEmbeddings(
        #     model="models/embedding-001",  # Google's text embedding model
        #     google_api_key=GOOGLE_API_KEY,
        # )

        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )

    except Exception as e:
        if is_quota_error(e):
            logger.warning("API quota exceeded. Vector search disabled.")
        else:
            logger.exception("Embeddings initialization failed")
        return None  # Allow system to continue without vector search