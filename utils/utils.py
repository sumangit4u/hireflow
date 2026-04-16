"""
Common utility functions for text processing, PDF handling, and error detection.
Centralized utilities used across the HireFlow project.
"""

import logging
import re
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_logger(name: str) -> logging.Logger:
    """Create standardized logger with timestamp formatting"""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s'
    )
    return logging.getLogger(name)

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

def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Split large text into overlapping chunks for processing"""
    logger = get_logger(__name__)
    
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]  # Split on paragraphs first, then lines, etc.
        )
        return splitter.split_text(text)
    except Exception as e:
        logger.error(f"Error splitting text: {e}")
        return [text]  # Return original text if splitting fails

def is_quota_error(error: Exception) -> bool:
    """Detect API quota/rate limit errors for graceful handling"""
    error_str = str(error).lower()
    return "429" in str(error) or "quota" in error_str or "rate limit" in error_str
