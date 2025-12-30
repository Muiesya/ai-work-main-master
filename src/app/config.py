import os
from functools import lru_cache
from typing import Optional

from pydantic import BaseModel, Field


class Settings(BaseModel):
    deepseek_api_key: Optional[str] = Field(
        default=None,
        description="API key for DeepSeek service; required for live generation.",
    )
    api_base_url: str = Field(default="https://api.deepseek.com", description="Base URL for DeepSeek API.")
    model: str = Field(default="deepseek-chat", description="Model name for completion API.")
    data_path: str = Field(default="data/drug_facts.json", description="Path to drug facts JSON file.")
    top_k: int = Field(default=3, description="Number of documents to retrieve for grounding.")
    safety_template: str = Field(
        default=(
            "You are a medication information assistant. Provide concise, non-personalized information "
            "grounded in the provided documents. Include a disclaimer that this is not medical advice "
            "and users should consult a healthcare professional. Do not invent dosing details if not provided."
        ),
        description="System prompt for the model.",
    )

    class Config:
        frozen = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"))
