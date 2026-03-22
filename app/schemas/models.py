from pydantic import BaseModel, Field
from typing import Optional, List


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        example="What is the population of Germany?",
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {"question": "What is the population of Germany?"},
                {"question": "What currency does Japan use?"},
                {"question": "What is the capital and population of Brazil?"},
            ]
        }


class AgentResponse(BaseModel):
    question: str
    answer: str
    country_detected: Optional[str] = None
    fields_requested: Optional[List[str]] = None