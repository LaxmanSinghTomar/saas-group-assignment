"""Application settings loader and persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict


DEFAULT_SETTINGS = {
    "top_k": 3,
    "semantic_weight": 0.7,
    "bm25_weight": 0.3,
    "temperature": 0.2,
}


@dataclass
class AppSettings:
    top_k: int = DEFAULT_SETTINGS["top_k"]
    semantic_weight: float = DEFAULT_SETTINGS["semantic_weight"]
    bm25_weight: float = DEFAULT_SETTINGS["bm25_weight"]
    temperature: float = DEFAULT_SETTINGS["temperature"]

    def update(self, data: Dict[str, Any]) -> None:
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SettingsManager:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._settings = AppSettings()
        self.load()

    @property
    def settings(self) -> AppSettings:
        return self._settings

    def load(self) -> None:
        if not self.path.exists():
            self.save()
            return
        with self.path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self._settings = AppSettings(**{**DEFAULT_SETTINGS, **data})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self._settings.to_dict(), f, ensure_ascii=False, indent=2)

    def update(self, data: Dict[str, Any]) -> AppSettings:
        self._settings.update(data)
        self.save()
        return self._settings


