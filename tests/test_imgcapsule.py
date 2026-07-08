from pathlib import Path

import imgcapsule as ic
from imgcapsule.capsule import Capsule


def write_ppm(path: Path, colors):
    width = len(colors)
    payload = bytearray()
    for color in colors:
        payload.extend(bytes(color))
    path.write_bytes(f"P6\n{width} 1\n255\n".encode("ascii") + bytes(payload))


def test_build_save_load_capsule(tmp_path):
    image = tmp_path / "sample.ppm"
    write_ppm(image, [(255, 0, 0), (0, 255, 0), (0, 0, 255)])

    capsule = ic.from_file(image)

    assert capsule.source.sha256
    assert capsule.source.media_type == "image/x-portable-pixmap"
    assert capsule.image.width == 3
    assert capsule.image.height == 1
    assert capsule.image.average_color is not None
    assert len(capsule.image.average_color) == 3
    assert capsule.image.perceptual_hash
    assert capsule.supports("near_duplicates")

    out = tmp_path / "sample.capsule.json"
    capsule.save(out)
    loaded = Capsule.load(out)
    assert loaded.source.sha256 == capsule.source.sha256


def test_store_search_and_duplicates(tmp_path):
    image = tmp_path / "sample.ppm"
    write_ppm(image, [(255, 0, 0), (250, 0, 0), (240, 0, 0)])

    db = tmp_path / "photos.icdb"
    with ic.Store(db) as store:
        capsule = store.add_path(image)
        results = store.search("portable")
        duplicates = store.near_duplicates(capsule.image.perceptual_hash)

    assert results
    assert results[0]["path"] == str(image)
    assert duplicates
    assert duplicates[0]["distance"] == 0
