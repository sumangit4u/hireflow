"""
LLM-powered resume parsing using LangChain and Pydantic.
Extracts structured data from resume text.
"""
import sys
sys.path.append(".")
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
import sys
sys.path.append("Hireflow")
from utils.config import GOOGLE_API_KEY, LLM_MODEL
from utils.schemas import Resume
from utils.utils import get_logger, clean_text, truncate_text

logger = get_logger(__name__)

class ResumeParser:

    def __init__(self):
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set in configuration.")
        
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GOOGLE_API_KEY
        )

        self.output_parser = PydanticOutputParser(pydantic_object=Resume)

        self.prompt_template = PromptTemplate(
            template = """You are an expert resume parser. Extract the following fields 
            from the resume text provided.
            
            Resume Text:
            {resume_text}

            {format_instructions}

            Return a valid JSON object that matches the Resume schema exactly.""",
            input_variables = ["resume_text"],
            partial_variables = {"format_instructions": self.output_parser.get_format_instructions()}
        )

    def parse_resume(self, resume_text: str, candidate_id: str) -> Dict[str, Any]:
        cleaned_text = clean_text(resume_text)
        truncated_text = truncate_text(cleaned_text)

        chain = self.prompt_template | self.llm | self.output_parser

        parsed_resume = chain.invoke({"resume_text": truncated_text})

        result = parsed_resume.model_dump()

        result.update(
            {
                "candidate_id": candidate_id,
                "raw_text": truncated_text
            }
        )
        return result

if __name__ == "__main__":
    sample_resume = """
    John Doe
    Email: Not present
    Phone: Not present
    Lives in New york.
    Experienced software engineer with three years in full-stack development. Having proficiency in python,
    jave and Azure.
    """
    parser = ResumeParser()
    parsed = parser.parse_resume(sample_resume, candidate_id="12345")
    print("Name:", parsed["name"])
    print("Email:", parsed["email"])
    print("Experience:", parsed["experience"])
    print("Skills:", parsed["skills"])
    print("Location:", parsed["location"])