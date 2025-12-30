import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


@dataclass(frozen=True)
class DrugRecord:
    generic_name: str
    brand_names: Sequence[str]
    uses: str
    dosage: str
    warnings: str
    side_effects: str
    sources: Sequence[str]
    last_updated: str

    @property
    def display_name(self) -> str:
        brands = f" (brands: {', '.join(self.brand_names)})" if self.brand_names else ""
        return f"{self.generic_name}{brands}"

    def to_text(self) -> str:
        return (
            f"Name: {self.display_name}\n"
            f"Uses: {self.uses}\n"
            f"Dosage: {self.dosage}\n"
            f"Warnings: {self.warnings}\n"
            f"Side effects: {self.side_effects}\n"
            f"Sources: {', '.join(self.sources)}\n"
            f"Last updated: {self.last_updated}"
        )


class KnowledgeBase:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.records: List[DrugRecord] = self._load_records()

    def _load_records(self) -> List[DrugRecord]:
        with self.data_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return [DrugRecord(**item) for item in data]

    def all_records(self) -> Sequence[DrugRecord]:
        return self.records
