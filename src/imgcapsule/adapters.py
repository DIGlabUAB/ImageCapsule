from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Protocol

from .capsule import Capsule


class ExtractorAdapter(Protocol):
    name: str
    version: str

    def available(self) -> bool:
        ...

    def apply(self, capsule: Capsule, image_path: Path) -> Capsule:
        ...


@dataclass
class TesseractOCRAdapter:
    name: str = "tesseract-ocr"
    version: str = "0.1.0"

    def available(self) -> bool:
        try:
            import pytesseract  # noqa: F401
            from PIL import Image  # noqa: F401
        except Exception:
            return False
        return True

    def apply(self, capsule: Capsule, image_path: Path) -> Capsule:
        if not self.available():
            capsule.extensions.setdefault("warnings", []).append("tesseract OCR adapter unavailable")
            return capsule
        import pytesseract
        from PIL import Image

        with Image.open(image_path) as img:
            text = pytesseract.image_to_string(img).strip()
        if text:
            capsule.merge_semantic(ocr_text=text, confidence={"ocr": 0.75})
            capsule.add_provenance(self.name, self.version, ["semantic.ocr_text", "semantic.confidence"], confidence=0.75)
        return capsule


def adapter_by_name(name: str) -> ExtractorAdapter:
    normalized = name.lower().strip()
    if normalized in {"ocr", "tesseract", "tesseract-ocr"}:
        return TesseractOCRAdapter()
    raise ValueError(f"unknown adapter: {name}")
