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
imgcapsule inspect photo.jpg.capsule.json

imgcapsule index ./photos --db photos.icdb
imgcapsule search photos.icdb "invoice"
imgcapsule duplicates photos.icdb photo.jpg
```

Python API:

```python
import imgcapsule as ic

capsule = ic.from_file("photo.jpg")
capsule.save("photo.jpg.capsule.json")

store = ic.Store("photos.icdb")
store.add_path("photo.jpg")
print(store.search("invoice"))
```

## What Is In A Capsule?

```json
{
  "schema_version": "0.1",
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
    "caption": null,
    "tags": [],
    "ocr_text": null,
    "embedding": null,
    "privacy_flags": []
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
