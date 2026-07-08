# imgcapsule

`imgcapsule` creates portable semantic sidecars for images.

An image file stores pixels. A capsule stores a compact, inspectable layer of meaning around those pixels: fingerprints, dimensions, perceptual hashes, previews, extracted text, tags, embeddings, privacy signals, and the provenance of the tools that produced them.

The core package is dependency-light and works locally. Richer extractors can be plugged in as optional adapters.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,vision]"
```

The `vision` extra enables richer local extraction through Pillow. Without it, the package still reads common image headers, file fingerprints, and PPM pixels.

## Quickstart

```bash
imgcapsule build photo.jpg
imgcapsule validate photo.jpg.capsule.json
imgcapsule inspect photo.jpg.capsule.json

imgcapsule index ./photos --db photos.icdb
imgcapsule search photos.icdb "invoice"
imgcapsule similar photos.icdb photo.jpg
imgcapsule duplicates photos.icdb photo.jpg
imgcapsule export photos.icdb capsules.jsonl
```

Python API:

```python
import imgcapsule as ic

capsule = ic.from_file("photo.jpg")
capsule.save("photo.jpg.capsule.json")

store = ic.Store("photos.icdb")
store.add_path("photo.jpg")
print(store.search("invoice"))
print(store.similar("photo.jpg"))
```

## What Is In A Capsule?

```json
{
  "schema_version": "0.2",
  "source": {
    "path": "photo.jpg",
    "sha256": "...",
    "media_type": "image/jpeg",
    "size_bytes": 12345
  },
  "image": {
    "width": 1024,
    "height": 768,
    "mode": "RGB",
    "average_color": [120, 98, 75],
    "perceptual_hash": "..."
  },
  "semantic": {
    "caption": "landscape image, 1024x768, average color rgb(120, 98, 75)",
    "tags": ["jpeg", "landscape"],
    "ocr_text": null,
    "embedding": [0.1, 0.2],
    "embedding_model": "imgcapsule-visual-v1",
    "embedding_dimensions": 29,
    "privacy_flags": ["has_exif"]
  },
  "provenance": [
    {
      "name": "core-file",
      "version": "0.1.0",
      "fields": ["source.sha256", "source.size_bytes"]
    }
  ]
}
```

## End-to-End Features

- Build portable `.capsule.json` files from images.
- Validate capsule schema and internal consistency.
- Store capsules in a local SQLite `.icdb`.
- Search by path, tags, caption, OCR text, and metadata.
- Find visually similar images using built-in local visual embeddings.
- Find near-duplicates using perceptual hashes.
- Export/import capsule stores as JSON Lines.
- Refresh capsules when extractors or models improve.
- Merge external model output into capsules with provenance.
- Run optional adapters such as OCR when dependencies are available.

## External Model Enrichment

`imgcapsule` does not force a single AI provider or model. Any OCR, captioning, safety, or embedding system can write JSON and merge it into the capsule:

```json
{
  "caption": "a scanned invoice on a white desk",
  "tags": ["invoice", "document", "receipt"],
  "ocr_text": "Invoice 123",
  "privacy_flags": ["contains_text"],
  "embedding": [0.12, -0.03, 0.44],
  "embedding_model": "my-vision-model-v1",
  "confidence": {"ocr": 0.91, "caption": 0.84}
}
```

```bash
imgcapsule enrich photo.jpg.capsule.json model-output.json
```

This keeps the image as the source of truth while making model outputs cached, versioned, inspectable, and replaceable.

## AI Model Providers

The base package works without AI dependencies. Real model adapters are optional and selected at runtime.

### Hugging Face Open-Source Models

Install the Hugging Face extra:

```bash
python -m pip install "imgcapsule[hf]"
```

Use default open-source models:

```bash
imgcapsule build photo.jpg --adapter hf-caption --adapter hf-clip --adapter hf-tags
imgcapsule index ./photos --db photos.icdb --adapter hf-caption --adapter hf-clip
```

Swap any model from Hugging Face:

```bash
imgcapsule build photo.jpg \
  --adapter hf-caption=Salesforce/blip-image-captioning-large \
  --adapter hf-clip=openai/clip-vit-large-patch14 \
  --adapter "hf-tags=openai/clip-vit-base-patch32|invoice,receipt,medical image,person"
```

Supported adapter names:

- `hf-caption=<model>`: image captioning model, default `Salesforce/blip-image-captioning-base`.
- `hf-clip=<model>`: image embedding model, default `openai/clip-vit-base-patch32`.
- `hf-tags=<model>|label1,label2`: zero-shot image labels, default `openai/clip-vit-base-patch32`.

If the optional dependencies are missing, the capsule still builds and records a warning under `extensions.warnings`.

### BYOK / Custom Vision Endpoint

Use any hosted model through an HTTP endpoint:

```bash
export IMGCAPSULE_BYOK_API_KEY="..."
export IMGCAPSULE_BYOK_MODEL="your-model-name"

imgcapsule build photo.jpg --adapter byok=https://your-endpoint.example/v1/image-capsule
```

Or set the endpoint once:

```bash
export IMGCAPSULE_BYOK_ENDPOINT="https://your-endpoint.example/v1/image-capsule"
imgcapsule index ./photos --db photos.icdb --adapter byok
```

The BYOK adapter sends:

```json
{
  "model": "your-model-name",
  "task": "image_capsule",
  "image_base64": "...",
  "media_type": "image/jpeg",
  "fields": ["caption", "tags", "ocr_text", "embedding", "privacy_flags"]
}
```

It accepts either a direct response:

```json
{
  "caption": "a scanned invoice",
  "tags": ["invoice", "document"],
  "ocr_text": "Invoice 123",
  "embedding": [0.1, 0.2, 0.3],
  "embedding_model": "my-model",
  "privacy_flags": ["contains_text"],
  "confidence": {"caption": 0.91}
}
```

or an OpenAI-compatible `choices[0].message.content` string containing JSON.

## Optional OCR

If `pytesseract`, Tesseract, and Pillow are installed, OCR can be enabled:

```bash
imgcapsule build photo.jpg --adapter ocr
imgcapsule index ./photos --db photos.icdb --adapter ocr
```

## Design Principles

- The original image remains the source of truth.
- The capsule is cached understanding: versioned, inspectable, and refreshable.
- Every generated field records provenance.
- The format is plain JSON first, so it is easy to debug and exchange.
- Search and duplicate detection should work without re-running extraction.

## Roadmap

- OCR adapter.
- CLIP/DINO embedding adapters.
- Privacy/safety adapter.
- Binary capsule format for large libraries.
- Browser and Node readers for capsule files.
