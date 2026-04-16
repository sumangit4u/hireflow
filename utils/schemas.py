"""Data schemas for HireFlow project."""

from dataclasses import dataclass, field as dc_field
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional


@dataclass
class SearchQuery:
    """Lightweight query context passed to the re-ranker and search functions."""
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
    candidate_id: str
    fit_score: int = Field(..., ge=0, le=100, description="Fit score (0-100)")
    strengths: List[str]
    gaps: List[str]
    risks: List[str] = Field(default_factory=list)
    summary: str

    evidence: Optional[dict] = Field(default_factory=dict)
