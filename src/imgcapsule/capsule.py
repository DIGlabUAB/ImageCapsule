from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "0.1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class SourceInfo:
    path: str
    sha256: str
    media_type: str
    size_bytes: int
    modified_at: Optional[str] = None


@dataclass
class ImageInfo:
    width: Optional[int] = None
    height: Optional[int] = None
    mode: Optional[str] = None
    average_color: Optional[List[int]] = None
    dominant_colors: List[List[int]] = field(default_factory=list)
    perceptual_hash: Optional[str] = None
    preview_data_uri: Optional[str] = None


@dataclass
class SemanticInfo:
    caption: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    ocr_text: Optional[str] = None
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    privacy_flags: List[str] = field(default_factory=list)


@dataclass
class ProvenanceRecord:
    name: str
    version: str
    fields: List[str]
    created_at: str = field(default_factory=utc_now)
    confidence: Optional[float] = None


@dataclass
class Capsule:
    source: SourceInfo
    image: ImageInfo = field(default_factory=ImageInfo)
    semantic: SemanticInfo = field(default_factory=SemanticInfo)
    provenance: List[ProvenanceRecord] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION
    created_at: str = field(default_factory=utc_now)
    refreshed_at: Optional[str] = None
    extensions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.write_text(self.to_json() + "\n", encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path) -> "Capsule":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Capsule":
        provenance = [
            ProvenanceRecord(**item) for item in data.get("provenance", [])
        ]
        return cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            created_at=data.get("created_at", utc_now()),
            refreshed_at=data.get("refreshed_at"),
            source=SourceInfo(**data["source"]),
            image=ImageInfo(**data.get("image", {})),
            semantic=SemanticInfo(**data.get("semantic", {})),
            provenance=provenance,
            extensions=data.get("extensions", {}),
        )

    def supports(self, capability: str) -> bool:
        if capability == "ocr":
            return bool(self.semantic.ocr_text)
        if capability == "semantic_search":
            return bool(self.semantic.embedding or self.semantic.caption or self.semantic.tags)
        if capability == "near_duplicates":
            return bool(self.image.perceptual_hash)
        if capability == "preview":
            return bool(self.image.preview_data_uri)
        return capability in self.extensions

    def needs_refresh(self, required_fields: Optional[List[str]] = None) -> bool:
        if not required_fields:
            return False
        current = self.to_dict()
        for field_path in required_fields:
            value: Any = current
            for part in field_path.split("."):
                if not isinstance(value, dict) or part not in value:
                    return True
                value = value[part]
            if value in (None, "", []):
                return True
        return False
