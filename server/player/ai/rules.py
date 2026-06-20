"""Rule loading and selection for AIPlayer prompts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from server.snapshot import StateSnapshot


@dataclass(frozen=True, slots=True)
class RuleBook:
    sections: dict[str, str]

    @classmethod
    def from_default(cls) -> "RuleBook":
        path = Path(__file__).with_name("rules.md")
        return cls.from_markdown(path.read_text(encoding="utf-8"))

    @classmethod
    def from_markdown(cls, text: str) -> "RuleBook":
        sections: dict[str, list[str]] = {}
        current: str | None = None
        for line in text.splitlines():
            if line.startswith("## "):
                current = line[3:].strip()
                sections[current] = []
            elif current is not None:
                sections[current].append(line)
        return cls({name: "\n".join(lines).strip() for name, lines in sections.items()})

    def select(self, snapshot: StateSnapshot) -> str:
        keys = ["common"]
        awaiting = snapshot.awaiting_action
        if awaiting == "bid":
            keys.append("bid")
        elif awaiting == "stir":
            keys.append("stir")
        elif awaiting == "discard":
            keys.extend(["discard", "scoring"])
        elif awaiting == "play":
            keys.append("play_lead" if _is_leading(snapshot) else "play_follow")
            keys.append("scoring")
        selected: list[str] = []
        for key in keys:
            section = self.sections.get(key)
            if section:
                selected.append(f"规则: {key}\n{section}")
        return "\n\n".join(selected)


def _is_leading(snapshot: StateSnapshot) -> bool:
    if snapshot.trick is None:
        return True
    lead_slot = snapshot.trick.slots[snapshot.trick.lead_player]
    return len(lead_slot.cards) == 0
