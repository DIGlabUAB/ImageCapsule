import tempfile
import json
import unittest
from pathlib import Path

import imgcapsule as ic
import imgcapsule.adapters as adapters_module
from imgcapsule.adapters import BYOKVisionAdapter, HuggingFaceCaptionAdapter, adapter_by_name
from imgcapsule.capsule import Capsule
from imgcapsule.cli import main


def write_ppm(path: Path, colors):
    width = len(colors)
    payload = bytearray()
    for color in colors:
        payload.extend(bytes(color))
    path.write_bytes(f"P6\n{width} 1\n255\n".encode("ascii") + bytes(payload))


class ImgCapsuleTests(unittest.TestCase):
    def test_build_save_load_capsule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "sample.ppm"
            write_ppm(image, [(255, 0, 0), (0, 255, 0), (0, 0, 255)])

            capsule = ic.from_file(image)

            self.assertTrue(capsule.source.sha256)
            self.assertEqual(capsule.source.media_type, "image/x-portable-pixmap")
            self.assertEqual(capsule.image.width, 3)
            self.assertEqual(capsule.image.height, 1)
            self.assertIsNotNone(capsule.image.average_color)
            self.assertEqual(len(capsule.image.average_color), 3)
            self.assertTrue(capsule.image.perceptual_hash)
            self.assertTrue(capsule.semantic.embedding)
            self.assertEqual(capsule.semantic.embedding_dimensions, len(capsule.semantic.embedding))
            self.assertTrue(capsule.semantic.caption)
            self.assertTrue(capsule.supports("near_duplicates"))
            self.assertTrue(capsule.supports("similarity_search"))
            self.assertEqual(capsule.validate(), [])

            out = root / "sample.capsule.json"
            capsule.save(out)
            loaded = Capsule.load(out)
            self.assertEqual(loaded.source.sha256, capsule.source.sha256)

    def test_store_search_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "sample.ppm"
            write_ppm(image, [(255, 0, 0), (250, 0, 0), (240, 0, 0)])

            db = root / "photos.icdb"
            with ic.Store(db) as store:
                capsule = store.add_path(image)
                results = store.search("portable")
                duplicates = store.near_duplicates(capsule.image.perceptual_hash)
                similar = store.similar(image)

            self.assertTrue(results)
            self.assertEqual(results[0]["path"], str(image))
            self.assertTrue(duplicates)
            self.assertEqual(duplicates[0]["distance"], 0)
            self.assertTrue(similar)
            self.assertGreater(similar[0]["score"], 0.99)

    def test_enrich_export_import_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "sample.ppm"
            write_ppm(image, [(255, 255, 255), (0, 0, 0), (255, 255, 255)])
            capsule_path = root / "sample.capsule.json"
            enrich_path = root / "enrich.json"
            enrich_path.write_text(
                '{"ocr_text":"Invoice 123","tags":["invoice","document"],"privacy_flags":["contains_text"],"confidence":{"ocr":0.9}}',
                encoding="utf-8",
            )

            self.assertEqual(main(["build", str(image), "-o", str(capsule_path)]), 0)
            self.assertEqual(main(["validate", str(capsule_path)]), 0)
            self.assertEqual(main(["enrich", str(capsule_path), str(enrich_path)]), 0)

            capsule = Capsule.load(capsule_path)
            self.assertEqual(capsule.semantic.ocr_text, "Invoice 123")
            self.assertIn("invoice", capsule.semantic.tags)
            self.assertIn("contains_text", capsule.semantic.privacy_flags)

            db = root / "photos.icdb"
            self.assertEqual(main(["index", str(image), "--db", str(db)]), 0)
            self.assertEqual(main(["search", str(db), "portable"]), 0)
            self.assertEqual(main(["similar", str(db), str(image)]), 0)
            out = root / "capsules.jsonl"
            self.assertEqual(main(["export", str(db), str(out)]), 0)
            imported = root / "imported.icdb"
            self.assertEqual(main(["import", str(imported), str(out)]), 0)

    def test_adapter_specs_and_missing_hf_are_safe(self):
        caption = adapter_by_name("hf-caption=example/model")
        self.assertIsInstance(caption, HuggingFaceCaptionAdapter)
        self.assertEqual(caption.model_id, "example/model")

        tags = adapter_by_name("hf-tags=example/tags|invoice,receipt")
        self.assertEqual(tags.model_id, "example/tags")
        self.assertEqual(tags.labels, ("invoice", "receipt"))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "sample.ppm"
            write_ppm(image, [(10, 20, 30), (20, 30, 40), (30, 40, 50)])
            capsule = ic.from_file(image, adapters=["hf-caption=missing/model"])
            if not caption.available():
                self.assertIn("warnings", capsule.extensions)

    def test_byok_adapter_enriches_capsule(self):
        received = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

            def read(self):
                body = {
                    "caption": "a test image from a custom model",
                    "tags": ["custom", "test"],
                    "ocr_text": "HELLO",
                    "embedding": [0.1, 0.2, 0.3],
                    "embedding_model": received["payload"]["model"],
                    "privacy_flags": ["contains_text"],
                    "confidence": {"caption": 0.99},
                }
                return json.dumps(body).encode("utf-8")

        def fake_urlopen(request, timeout=60):
            received["payload"] = json.loads(request.data.decode("utf-8"))
            received["timeout"] = timeout
            return FakeResponse()

        original_urlopen = adapters_module.urlopen
        adapters_module.urlopen = fake_urlopen
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                image = root / "sample.ppm"
                write_ppm(image, [(10, 20, 30), (20, 30, 40), (30, 40, 50)])
                capsule = ic.from_file(image, adapters=["byok=http://example.test/capsule"])
                self.assertEqual(capsule.semantic.caption, "a test image from a custom model")
                self.assertIn("custom", capsule.semantic.tags)
                self.assertEqual(capsule.semantic.ocr_text, "HELLO")
                self.assertEqual(capsule.semantic.embedding_model, "vision-model")
                self.assertIn("contains_text", capsule.semantic.privacy_flags)
                self.assertEqual(received["payload"]["task"], "image_capsule")
        finally:
            adapters_module.urlopen = original_urlopen


if __name__ == "__main__":
    unittest.main()
