from typing import Any, Dict, List, Tuple

from openai import OpenAI

from .config import get_settings
from .knowledge_base import DrugRecord, KnowledgeBase
from .retriever import Retriever


class QAEngine:
    def __init__(self, kb: KnowledgeBase | None = None, retriever: Retriever | None = None):
        self.settings = get_settings()
        self.kb = kb or KnowledgeBase(self.settings.data_path)
        self.retriever = retriever or Retriever(self.kb)
        if not self.settings.deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for generation.")
        self.client = OpenAI(api_key=self.settings.deepseek_api_key, base_url=self.settings.api_base_url)

    def build_messages(self, query: str, context: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": self.settings.safety_template},
            {
                "role": "system",
                "content": (
                    "Grounding documents:\n" + context
                ),
            },
            {"role": "user", "content": query},
        ]

    def generate_answer(self, query: str) -> Dict[str, Any]:
        retrieved: List[Tuple[DrugRecord, float]] = self.retriever.retrieve(query, k=self.settings.top_k)
        context = self.retriever.format_context(retrieved)
        messages = self.build_messages(query, context)

        response = self.client.chat.completions.create(
            model=self.settings.model,
            messages=messages,
            temperature=0.2,
        )
        answer = response.choices[0].message.content

        return {
            "question": query,
            "answer": answer,
            "sources": [record.display_name for record, _ in retrieved],
            "last_updated": [record.last_updated for record, _ in retrieved],
        }
