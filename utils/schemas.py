"""Data schemas for HireFlow project."""

from dataclasses import dataclass, field as dc_field
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional


@dataclass
class SearchQuery:
    """Lightweight query context passed to search and evaluation."""
    title: str
    text: str
    required_skills: List[str] = dc_field(default_factory=list)


class Resume(BaseModel):
    candidate_id: str = Field(..., description="Unique ID for candidate")
    name: str = Field(..., description="Candidate full name")
    email: Optional[EmailStr] = Field(None, description="Candidate email")
    phone: Optional[str] = Field(None, description="Candidate phone")
    location: Optional[str] = Field(None, description="Candidate location")

    text: str = Field(..., description="Raw text of resume")
    skills: List[str] = Field(default_factory=list, description="skills")
    experience: Optional[int] = Field(None, description="Total years of experience")


class CandidateEvaluation(BaseModel):
    """Qualitative LLM assessment of a candidate.

    Deliberately carries no numeric score. Ranking is expressed by the three
    retrieval scores (BM25, vector, and their RRF combination); adding a
    fourth, differently-derived number here only obscured what those mean.
    """
    candidate_id: str
    strengths: List[str]
    gaps: List[str]
    risks: List[str] = Field(default_factory=list)
    summary: str
