from pydantic import BaseModel, Field
from typing import List


class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing")
    superfluous:str = Field(description="Critique of what is superfluous")

class AnswerQuestion(BaseModel):
    """Answer the qustion."""

    answer: str = Field(
        description="~250 words detailed answer to the question.")
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")

class RevisedAnswer(AnswerQuestion):
    """Revise your original answer to your question."""

    references: list[str] = Field(
        description= "Citations motivating your updated answer."
    )