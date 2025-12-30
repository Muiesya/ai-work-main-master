from app.knowledge_base import KnowledgeBase
from app.retriever import Retriever


def test_retriever_returns_ranked_results():
    kb = KnowledgeBase("data/drug_facts.json")
    retriever = Retriever(kb)

    results = retriever.retrieve("fever relief", k=2)

    assert results, "Expected at least one retrieval result"
    records = [rec.generic_name for rec, _ in results]
    assert "acetaminophen" in records or "ibuprofen" in records


def test_format_context_contains_relevance_scores():
    kb = KnowledgeBase("data/drug_facts.json")
    retriever = Retriever(kb)
    results = retriever.retrieve("antibiotic for infections", k=1)

    context = retriever.format_context(results)

    assert "Relevance" in context
    assert any(rec.generic_name in context for rec, _ in results)
