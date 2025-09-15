from __future__ import annotations
from typing import List, Dict, Any

class ShortTerm:
    def __init__(self):
        self.events: List[Dict[str, Any]] = []
    def log(self, kind: str, data: Dict[str, Any]):
        self.events.append({"kind": kind, **data})
    def to_markdown(self) -> str:
        lines = ["## Trace\n"]
        for e in self.events:
            lines.append(f"- **{e['kind']}**: {{ {', '.join(f'{k}:{v}' for k,v in e.items() if k!='kind')} }}")
        return "\n".join(lines)

class LongTerm:
    # Placeholder for future vector store of past incidents
    def __init__(self, name: str = "incidents"):
        self.name = name
    def upsert(self, summary_id: str, text: str, meta: Dict[str, Any]):
        pass
