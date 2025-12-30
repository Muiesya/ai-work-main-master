import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

from .knowledge_base import DrugRecord, KnowledgeBase


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9]+", text.lower()) if t]


class Retriever:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.documents = [record.to_text() for record in kb.all_records()]
        self.doc_tokens = [_tokenize(doc) for doc in self.documents]
        self.doc_freqs: Dict[str, int] = self._compute_document_frequencies()
        self.idf: Dict[str, float] = self._compute_idf()
        self.doc_vectors: List[Dict[str, float]] = [
            self._compute_tf_idf(tokens) for tokens in self.doc_tokens
        ]

    def _compute_document_frequencies(self) -> Dict[str, int]:
        df: Dict[str, int] = {}
        for tokens in self.doc_tokens:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
        return df

    def _compute_idf(self) -> Dict[str, float]:
        total_docs = len(self.doc_tokens)
        return {
            term: math.log((1 + total_docs) / (1 + df)) + 1.0
            for term, df in self.doc_freqs.items()
        }

    def _compute_tf_idf(self, tokens: List[str]) -> Dict[str, float]:
        counts = Counter(tokens)
        total = sum(counts.values()) or 1
        return {term: (count / total) * self.idf.get(term, 0.0) for term, count in counts.items()}

    def _cosine_similarity(self, vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        common_terms = set(vec_a) & set(vec_b)
        numerator = sum(vec_a[t] * vec_b[t] for t in common_terms)
        denom_a = math.sqrt(sum(v * v for v in vec_a.values()))
        denom_b = math.sqrt(sum(v * v for v in vec_b.values()))
        if denom_a == 0 or denom_b == 0:
            return 0.0
        return numerator / (denom_a * denom_b)

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[DrugRecord, float]]:
        if not query:
            return []
        query_vec = self._compute_tf_idf(_tokenize(query))
        scores = [self._cosine_similarity(query_vec, doc_vec) for doc_vec in self.doc_vectors]
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            (self.kb.records[idx], float(scores[idx]))
            for idx in ranked_indices
            if scores[idx] > 0
        ]

    def format_context(self, results: Sequence[Tuple[DrugRecord, float]]) -> str:
        blocks = []
        for record, score in results:
            blocks.append(
                f"[Relevance: {score:.2f}]\n{record.to_text()}"
            )
        return "\n\n".join(blocks)
