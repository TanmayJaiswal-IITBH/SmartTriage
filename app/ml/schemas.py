from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Annotated

class IssueText(BaseModel):
    """
    Input payload from the Integrator.
    Contains the sanitized text of a newly opened GitHub Issue.
    """
    model_config = ConfigDict(str_strip_whitespace=True)
    
    repo_name: Annotated[str, Field(min_length=1)]
    issue_number: int
    title: Annotated[str, Field(min_length=1)]
    body: str = ""

    @property
    def embedding_text(self) -> str:
        """Fuses title and body to feed into the sentence-transformer."""
        return f"{self.title}\n{self.body}"

class DuplicateResult(BaseModel):
    """
    Output payload sent back to the Integrator.
    Dictates whether the API should close the issue as a duplicate.
    """
    is_duplicate: bool
    similarity_score: Annotated[
        float, 
        Field(ge=0.0, le=1.0, description="Cosine similarity score from 0.0 to 1.0")
    ]
    duplicate_of_number: Optional[int] = Field(
        default=None, 
        description="The issue number this duplicates"
    )

class PRMetadata(BaseModel):
    """
    Input payload from the Integrator.
    Contains the author and the list of files modified in a new PR.
    """
    repo_name: Annotated[str, Field(min_length=1)]
    pr_number: int
    author_username: str
    changed_files: List[str]

class ReviewerScore(BaseModel):
    """
    Represents the calculated expertise of a single developer for a PR.
    """
    github_username: str
    relevance_score: Annotated[
        float, 
        Field(ge=0.0, description="Calculated weight based on commit history")
    ]

class ReviewerRecommendation(BaseModel):
    """
    Output payload sent back to the Integrator.
    Contains the top developers the API should request reviews from.
    """
    suggested_reviewers: List[ReviewerScore]