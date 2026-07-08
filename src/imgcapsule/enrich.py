from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .capsule import Capsule


ENRICHER_VERSION = "0.1.0"


def enrich_from_mapping(capsule: Capsule, data: Dict[str, Any], *, name: str = "manual") -> Capsule:
    tags = data.get("tags")
    if isinstance(tags, str):
        tags = [item.strip() for item in tags.split(",") if item.strip()]
    privacy_flags = data.get("privacy_flags")
    if isinstance(privacy_flags, str):
        privacy_flags = [item.strip() for item in privacy_flags.split(",") if item.strip()]
    embedding = data.get("embedding")
    if embedding is not None:
        embedding = [float(value) for value in embedding]

    capsule.merge_semantic(
        caption=data.get("caption"),
        tags=tags,
        ocr_text=data.get("ocr_text"),
        embedding=embedding,
        embedding_model=data.get("embedding_model"),
        privacy_flags=privacy_flags,
        confidence=data.get("confidence") if isinstance(data.get("confidence"), dict) else None,
    )
    fields = []
    for key in ["caption", "tags", "ocr_text", "embedding", "embedding_model", "privacy_flags", "confidence"]:
        if key in data:
            fields.append(f"semantic.{key}")
    capsule.add_provenance(name=name, version=ENRICHER_VERSION, fields=fields or ["semantic"])
    return capsule


def enrich_from_json_file(capsule: Capsule, path: str | Path, *, name: str = "manual-json") -> Capsule:
    import json

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("enrichment JSON must contain an object")
    return enrich_from_mapping(capsule, data, name=name)
