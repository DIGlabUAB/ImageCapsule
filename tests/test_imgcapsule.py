import tempfile
import unittest
from pathlib import Path

import imgcapsule as ic
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


if __name__ == "__main__":
    unittest.main()
