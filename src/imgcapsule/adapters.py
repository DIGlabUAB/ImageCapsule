from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple
from urllib.request import Request, urlopen

from .capsule import Capsule


DEFAULT_HF_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
DEFAULT_HF_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_HF_TAG_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_HF_TAG_LABELS = (
    "document",
    "receipt",
    "screenshot",
    "person",
    "landscape",
    "medical image",
    "food",
    "product",
    "text",
)


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


@dataclass
class HuggingFaceCaptionAdapter:
    model_id: str = DEFAULT_HF_CAPTION_MODEL
    name: str = "huggingface-caption"
    version: str = "0.1.0"

    def available(self) -> bool:
        try:
            import transformers  # noqa: F401
            from PIL import Image  # noqa: F401
        except Exception:
            return False
        return True

    def apply(self, capsule: Capsule, image_path: Path) -> Capsule:
        if not self.available():
            _warn(capsule, "huggingface caption adapter unavailable; install imgcapsule[hf]")
            return capsule
        from PIL import Image
        from transformers import pipeline

        generator = pipeline("image-to-text", model=self.model_id)
        with Image.open(image_path) as img:
            result = generator(img)
        caption = _first_generated_text(result)
        if caption:
            capsule.merge_semantic(
                caption=caption,
                tags=_keywords(caption),
                confidence={"caption": 0.8},
            )
            capsule.add_provenance(
                self.name,
                self.model_id,
                ["semantic.caption", "semantic.tags", "semantic.confidence"],
                confidence=0.8,
            )
        return capsule


@dataclass
class HuggingFaceCLIPAdapter:
    model_id: str = DEFAULT_HF_CLIP_MODEL
    name: str = "huggingface-clip"
    version: str = "0.1.0"

    def available(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            from PIL import Image  # noqa: F401
        except Exception:
            return False
        return True

    def apply(self, capsule: Capsule, image_path: Path) -> Capsule:
        if not self.available():
            _warn(capsule, "huggingface CLIP adapter unavailable; install imgcapsule[hf]")
            return capsule
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor

        processor = CLIPProcessor.from_pretrained(self.model_id)
        model = CLIPModel.from_pretrained(self.model_id)
        with Image.open(image_path) as img:
            inputs = processor(images=img.convert("RGB"), return_tensors="pt")
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        embedding = [float(value) for value in features[0].tolist()]
        capsule.merge_semantic(
            embedding=embedding,
            embedding_model=self.model_id,
            confidence={"embedding": 0.9},
        )
        capsule.add_provenance(
            self.name,
            self.model_id,
            ["semantic.embedding", "semantic.embedding_model", "semantic.embedding_dimensions", "semantic.confidence"],
            confidence=0.9,
        )
        return capsule


@dataclass
class HuggingFaceZeroShotAdapter:
    model_id: str = DEFAULT_HF_TAG_MODEL
    labels: Tuple[str, ...] = DEFAULT_HF_TAG_LABELS
    name: str = "huggingface-zero-shot"
    version: str = "0.1.0"

    def available(self) -> bool:
        try:
            import transformers  # noqa: F401
            from PIL import Image  # noqa: F401
        except Exception:
            return False
        return True

    def apply(self, capsule: Capsule, image_path: Path) -> Capsule:
        if not self.available():
            _warn(capsule, "huggingface zero-shot adapter unavailable; install imgcapsule[hf]")
            return capsule
        from PIL import Image
        from transformers import pipeline

        classifier = pipeline("zero-shot-image-classification", model=self.model_id)
        with Image.open(image_path) as img:
            result = classifier(img, candidate_labels=list(self.labels))
        tags = [
            item["label"]
            for item in result
            if float(item.get("score", 0.0)) >= 0.2
        ][:5]
        confidence = float(result[0].get("score", 0.0)) if result else 0.0
        if tags:
            capsule.merge_semantic(tags=tags, confidence={"tags": confidence})
            capsule.add_provenance(
                self.name,
                self.model_id,
                ["semantic.tags", "semantic.confidence"],
                confidence=confidence,
            )
        return capsule


@dataclass
class BYOKVisionAdapter:
    endpoint: Optional[str] = None
    model_id: Optional[str] = None
    name: str = "byok-vision"
    version: str = "0.1.0"

    def available(self) -> bool:
        return bool(self.endpoint or os.environ.get("IMGCAPSULE_BYOK_ENDPOINT"))

    def apply(self, capsule: Capsule, image_path: Path) -> Capsule:
        endpoint = self.endpoint or os.environ.get("IMGCAPSULE_BYOK_ENDPOINT")
        if not endpoint:
            _warn(capsule, "BYOK adapter unavailable; set IMGCAPSULE_BYOK_ENDPOINT or pass byok=<url>")
            return capsule
        model_id = self.model_id or os.environ.get("IMGCAPSULE_BYOK_MODEL", "vision-model")
        api_key = os.environ.get("IMGCAPSULE_BYOK_API_KEY")
        payload = {
            "model": model_id,
            "task": "image_capsule",
            "image_base64": base64.b64encode(image_path.read_bytes()).decode("ascii"),
            "media_type": capsule.source.media_type,
            "fields": ["caption", "tags", "ocr_text", "embedding", "privacy_flags"],
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        try:
            with urlopen(request, timeout=float(os.environ.get("IMGCAPSULE_BYOK_TIMEOUT", "60"))) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            _warn(capsule, f"BYOK adapter failed: {exc}")
            return capsule

        data = _unwrap_byok_response(data)
        capsule.merge_semantic(
            caption=data.get("caption"),
            tags=data.get("tags"),
            ocr_text=data.get("ocr_text"),
            embedding=data.get("embedding"),
            embedding_model=data.get("embedding_model") or model_id,
            privacy_flags=data.get("privacy_flags"),
            confidence=data.get("confidence") if isinstance(data.get("confidence"), dict) else None,
        )
        capsule.add_provenance(
            self.name,
            model_id,
            ["semantic.caption", "semantic.tags", "semantic.ocr_text", "semantic.embedding", "semantic.privacy_flags"],
        )
        return capsule


def adapter_by_name(name: str) -> ExtractorAdapter:
    normalized, value = _split_adapter_spec(name)
    if normalized in {"ocr", "tesseract", "tesseract-ocr"}:
        return TesseractOCRAdapter()
    if normalized in {"hf-caption", "huggingface-caption", "caption"}:
        return HuggingFaceCaptionAdapter(model_id=value or DEFAULT_HF_CAPTION_MODEL)
    if normalized in {"hf-clip", "huggingface-clip", "clip", "embedding"}:
        return HuggingFaceCLIPAdapter(model_id=value or DEFAULT_HF_CLIP_MODEL)
    if normalized in {"hf-tags", "huggingface-tags", "tags", "zero-shot"}:
        model_id, labels = _split_model_and_labels(value)
        return HuggingFaceZeroShotAdapter(
            model_id=model_id or DEFAULT_HF_TAG_MODEL,
            labels=tuple(labels) if labels else DEFAULT_HF_TAG_LABELS,
        )
    if normalized in {"byok", "custom", "http"}:
        return BYOKVisionAdapter(endpoint=value or None)
    raise ValueError(f"unknown adapter: {name}")


def _split_adapter_spec(spec: str) -> Tuple[str, str]:
    if "=" in spec:
        name, value = spec.split("=", 1)
    else:
        name, value = spec, ""
    return name.lower().strip(), value.strip()


def _split_model_and_labels(value: str) -> Tuple[str, List[str]]:
    if not value:
        return "", []
    if "|" not in value:
        return value, []
    model_id, labels = value.split("|", 1)
    return model_id.strip(), [label.strip() for label in labels.split(",") if label.strip()]


def _first_generated_text(result: Any) -> Optional[str]:
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            return first.get("generated_text") or first.get("caption") or first.get("text")
    if isinstance(result, dict):
        return result.get("generated_text") or result.get("caption") or result.get("text")
    return None


def _keywords(text: str) -> List[str]:
    stop = {"the", "and", "with", "from", "that", "this", "image", "photo", "picture"}
    words = []
    for raw in text.lower().replace(",", " ").replace(".", " ").split():
        word = "".join(char for char in raw if char.isalnum() or char in {"-", "_"})
        if len(word) >= 3 and word not in stop:
            words.append(word)
    return list(dict.fromkeys(words))[:8]


def _unwrap_byok_response(data: Dict[str, Any]) -> Dict[str, Any]:
    if "capsule" in data and isinstance(data["capsule"], dict):
        return data["capsule"]
    if "choices" in data and data["choices"]:
        content = data["choices"][0].get("message", {}).get("content")
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return {"caption": content}
    return data


def _warn(capsule: Capsule, message: str) -> None:
    capsule.extensions.setdefault("warnings", []).append(message)
