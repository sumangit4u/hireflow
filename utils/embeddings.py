"""Google Generative AI embedding model initialization.

Handles API quota errors and ensures an asyncio event loop exists when
called from threads (Streamlit runs app code in a worker thread).
"""

import asyncio
import logging
from contextlib import contextmanager
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


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@contextmanager
def _quiet_progress_bars():
    """Silence HuggingFace/transformers tqdm bars for the duration of a block.

    Best-effort: these live in different places across library versions, so
    every step is guarded and a failure just means the bar still shows.
    """
    restore = []
    try:
        try:
            from huggingface_hub.utils import (
                disable_progress_bars, enable_progress_bars,
            )
            disable_progress_bars()
            restore.append(enable_progress_bars)
        except Exception:
            pass
        try:
            from transformers.utils import logging as hf_logging
            hf_logging.disable_progress_bar()
            restore.append(hf_logging.enable_progress_bar)
        except Exception:
            pass
        yield
    finally:
        for restore_fn in restore:
            try:
                restore_fn()
            except Exception:
                pass


def get_embeddings() -> Optional[HuggingFaceEmbeddings]:
    """Load the local sentence-transformers embedding model.

    Runs on this machine — no API key, no quota, no per-call cost. Gemini is
    used elsewhere for parsing and re-ranking, but never for embeddings.

    Tries the local cache first. sentence-transformers otherwise revalidates
    the model against huggingface.co on *every* load, which means a dozen HTTP
    round-trips (and a dozen log lines) each time the app starts, even though
    the files are already on disk. The download path is kept as a fallback so
    a cold cache still works.

    Returns `None` if embeddings cannot be created, so the app can degrade to
    BM25-only search instead of failing outright.
    """
    try:
        _ensure_event_loop()

        # return GoogleGenerativeAIEmbeddings(
        #     model="models/embedding-001",  # Google's text embedding model
        #     google_api_key=GOOGLE_API_KEY,
        # )

        try:
            # Cached load: nothing to report, so suppress the weight-loading
            # progress bar. It is left enabled on the download path below,
            # where a first-time user genuinely wants to see progress.
            with _quiet_progress_bars():
                return HuggingFaceEmbeddings(
                    model_name=MODEL_NAME,
                    model_kwargs={"local_files_only": True},
                )
        except Exception:
            logger.info(
                f"{MODEL_NAME} not in the local cache — downloading it once."
            )
            return HuggingFaceEmbeddings(model_name=MODEL_NAME)

    except Exception as e:
        if is_quota_error(e):
            logger.warning("API quota exceeded. Vector search disabled.")
        else:
            logger.exception("Embeddings initialization failed")
        return None  # Allow system to continue without vector search