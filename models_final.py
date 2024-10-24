from pydantic import BaseModel
from enum import Enum
from typing import Optional, List

class ContentType(Enum):
    TWEET = "tweet"
    EMAIL = "email"
    TEXT_MESSAGE = "text_message"
    LINKEDIN_POST = "linkedin_post"
    INSTAGRAM_CAPTION = "instagram_caption"

class WritingContent(BaseModel):
    content: str
    tone: str
    word_count: Optional[int]

class EvaluationScore(BaseModel):
    reasoning: str
    score: float  # 0-10 scale
    suggestions: List[str]

class ContentEvaluation(BaseModel):
    clarity: EvaluationScore
    engagement: EvaluationScore
    tone_consistency: EvaluationScore
    originality: EvaluationScore
    platform_fit: EvaluationScore
    timestamp: str

class ContentRewrite(BaseModel):
    original_content: str
    improved_content: str
    changes_made: List[str]
    improvement_focus: List[str]
