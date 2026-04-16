"""
Document loading and processing for resumes.
Converts PDF files to LangChain Document objects with rich metadata.
"""

import os
from pathlib import Path
from langchain.schema import Document
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.utils import get_logger, load_pdf, split_text

logger = get_logger(__name__)


def _try_parse_resume(text: str, candidate_id: str) -> dict:
    """Attempt to parse resume text with ResumeParser (Gemini).

    Returns a dict with keys like 'skills', 'location', 'experience', 'name'.
    If the parser is unavailable or fails, returns an empty dict so ingestion
    can still proceed with basic metadata.
    """
    try:
        from core.parsing import ResumeParser
        parser = ResumeParser()
        parsed = parser.parse_resume(text, candidate_id)
        return parsed
    except Exception as e:
        logger.warning(f"Resume parsing failed for candidate {candidate_id}: {e}")
        return {}

def load_resumes(directory: str) -> list[Document]:
    """Load all resume PDFs from directory, parse them with Gemini, and
    return Document objects with rich metadata (skills, location, experience)."""
    resumes = []

    if not os.listdir(directory):
        return resumes
    
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if file.lower().endswith('.pdf'):
            try:
                text = load_pdf(file_path)
                if not text:
                    continue

                candidate_id = f"c_{Path(file).stem}"
                fallback_name = Path(file).stem.replace('_', ' ').title()

                parsed = _try_parse_resume(text, candidate_id)

                doc = Document(
                    page_content = text,
                    metadata = {
                        "source": file_path,
                        "filename":file,
                        "candidate_id": candidate_id,
                        "name": parsed.get('name', fallback_name),
                        "skills": parsed.get('skills', []),
                        "location": parsed.get('location', 'Unknown'),
                        "experience": parsed.get('experience', 0)
                    }
                )
                resumes.append(doc)
            except Exception as e:
                logger.error(f"Failed to load resume {file_path}: {e}")
                continue
    return resumes

class DocumentProcessor:
    """Legacy document processor class - kept for backward compatibility"""

    def __init__(self,chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def load_pdf(self, file_path: str):
        """Load a PDF file and return as a Document object"""
        return load_pdf(file_path)
    
    def split_text(self, text: str):
        """Create chunks from the resume text"""
        return split_text(text, chunk_size=self.chunk_size, chunk_overlap=self.overlap)
    
    def process_resume_pdf(self, file_path:str):
        """Process a resume PDF file and return Document object"""
        text = self.load_pdf(file_path)
        # chunks = self.split_text(text)
        # docs = []
        # if text:
        #     for chunk in chunks:
        #         doc = Document(page_content=chunk, metadata={"source": f"{Path(file_path).stem}"})
        #         docs.append(doc)
        #     return docs 
        if text:
            return Document(
                page_content=text, 
                metadata={"source": f"{Path(file_path).stem}"}
            )  
        return None

def _default_data_dirs():
    """Return sensible default directories relative to project root."""
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    resumes_dir = project_root / "data" / "resumes"
    return str(resumes_dir)


def _print_sample_documents(docs, label: str, max_display: int = 3):
    print(f"\n{label}: {len(docs)} found")
    for i, doc in enumerate(docs[:max_display]):
        meta = getattr(doc, 'metadata', {}) or {}
        print(f"[{i+1}] id: {meta.get('candidate_id', meta.get('jd_id', 'n/a'))} | filename: {meta.get('filename', meta.get('title', 'n/a'))}")
        snippet = (getattr(doc, 'page_content', '') or '')[:200]
        print(f"    snippet: {snippet.replace('\n', ' ')[:180]}...\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Ingestion demo: load resumes')
    default_resumes = _default_data_dirs()
    parser.add_argument('--resumes-dir', type=str, default=default_resumes, help='Path to resumes directory (PDFs)')
    parser.add_argument('--show', action='store_true', help='Print a short sample of loaded documents')

    args = parser.parse_args()

    print(f"Using resumes dir: {args.resumes_dir}")

    try:
        resumes = load_resumes(args.resumes_dir)
        # print(resumes[0])

        if args.show:
            _print_sample_documents(resumes, 'Resumes')
        else:
            print(f"Loaded {len(resumes)} resumes.")

        # dp = DocumentProcessor()
        # if resumes:
        #     sample_path = resumes[0].metadata.get('source')
        #     if sample_path and hasattr(dp, 'process_resume_pdf'):
        #         proc_doc = dp.process_resume_pdf(sample_path)
        #         if proc_doc:
        #             print(proc_doc)
        #             print('\nDocumentProcessor processed a resume file and returned a Document object (sample).')
        #         else:
        #             print('\nDocumentProcessor could not extract text from the sample resume.')

    except Exception as e:
        print(f"Ingestion demo failed: {e}")
