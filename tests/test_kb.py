from pathlib import Path

from app.knowledge_base import KnowledgeBase


def test_loads_sample_records():
    kb = KnowledgeBase("data/drug_facts.json")
    assert len(kb.all_records()) >= 3
    names = {rec.generic_name for rec in kb.all_records()}
    assert {"acetaminophen", "ibuprofen", "amoxicillin"}.issubset(names)


def test_display_name_includes_brands():
    kb = KnowledgeBase("data/drug_facts.json")
    acetaminophen = next(rec for rec in kb.all_records() if rec.generic_name == "acetaminophen")
    assert "Tylenol" in acetaminophen.display_name
    assert acetaminophen.display_name.startswith("acetaminophen")
