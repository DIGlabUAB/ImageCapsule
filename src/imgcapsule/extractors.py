from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import mimetypes
from pathlib import Path
import struct
from typing import Iterable, List, Optional, Tuple

from .adapters import adapter_by_name
from .capsule import Capsule, ImageInfo, ProvenanceRecord, SemanticInfo, SourceInfo
from .similarity import normalize


EXTRACTOR_VERSION = "0.1.0"


def from_file(path: str | Path, *, adapters: Optional[Iterable[str]] = None) -> Capsule:
    image_path = Path(path)
    data = image_path.read_bytes()
    stat = image_path.stat()
    media_type = _guess_media_type(image_path, data)
    modified_at = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat().replace("+00:00", "Z")

    source = SourceInfo(
        path=str(image_path),
        sha256=hashlib.sha256(data).hexdigest(),
        media_type=media_type,
        size_bytes=len(data),
        modified_at=modified_at,
    )
    image = _extract_with_pillow(image_path)
    if image is None:
        image = _extract_core(data, media_type)
    embedding = _visual_embedding(image)
    tags = _basic_tags(media_type, image)
    caption = _heuristic_caption(image, tags)
    privacy_flags = _privacy_flags(image)

    capsule = Capsule(
        source=source,
        image=image,
        semantic=SemanticInfo(
            caption=caption,
            tags=tags,
            embedding=embedding,
            embedding_model="imgcapsule-visual-v1",
            embedding_dimensions=len(embedding),
            privacy_flags=privacy_flags,
        ),
        provenance=[
            ProvenanceRecord(
                name="core-file",
                version=EXTRACTOR_VERSION,
                fields=[
                    "source.path",
                    "source.sha256",
                    "source.media_type",
                    "source.size_bytes",
                    "source.modified_at",
                ],
            ),
            ProvenanceRecord(
                name="core-image",
                version=EXTRACTOR_VERSION,
                fields=[
                    "image.width",
                    "image.height",
                    "image.mode",
                    "image.average_color",
                    "image.perceptual_hash",
                    "image.preview_data_uri",
                    "image.exif",
                    "semantic.tags",
                    "semantic.caption",
                    "semantic.embedding",
                    "semantic.embedding_model",
                    "semantic.embedding_dimensions",
                    "semantic.privacy_flags",
                ],
            ),
        ],
    )
    for adapter_name in adapters or []:
        adapter = adapter_by_name(adapter_name)
        capsule = adapter.apply(capsule, image_path)
    return capsule


def _extract_with_pillow(path: Path) -> Optional[ImageInfo]:
    try:
        from PIL import Image
    except Exception:
        return None

    try:
        with Image.open(path) as img:
            original = img.copy()
            rgb = original.convert("RGB")
            small = rgb.resize((1, 1))
            average = list(small.getpixel((0, 0)))
            colors = _dominant_colors_from_pillow(rgb)
            phash = _dhash_from_pillow(rgb)
            preview = _preview_from_pillow(rgb)
            exif = _exif_from_pillow(original)
            return ImageInfo(
                width=original.width,
                height=original.height,
                mode=original.mode,
                average_color=average,
                dominant_colors=colors,
                perceptual_hash=phash,
                preview_data_uri=preview,
                exif=exif,
            )
    except Exception:
        return None


def _dominant_colors_from_pillow(img) -> List[List[int]]:
    small = img.resize((32, 32))
    colors = small.quantize(colors=5).convert("RGB").getcolors(maxcolors=1024)
    if not colors:
        return []
    ordered = sorted(colors, key=lambda item: item[0], reverse=True)
    return [list(pixel) for _, pixel in ordered[:5]]


def _dhash_from_pillow(img) -> str:
    gray = img.convert("L").resize((9, 8))
    pixels = list(gray.getdata())
    bits = []
    for row in range(8):
        offset = row * 9
        for col in range(8):
            bits.append(1 if pixels[offset + col] > pixels[offset + col + 1] else 0)
    return _bits_to_hex(bits)


def _preview_from_pillow(img) -> str:
    from io import BytesIO

    preview = img.copy()
    preview.thumbnail((96, 96))
    buffer = BytesIO()
    preview.save(buffer, format="JPEG", quality=65, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _extract_core(data: bytes, media_type: str) -> ImageInfo:
    width, height = _read_dimensions(data, media_type)
    average = None
    phash = None
    mode = None
    if data.startswith(b"P6") or data.startswith(b"P3"):
        width, height, pixels = _read_ppm(data)
        if pixels:
            mode = "RGB"
            average = [
                sum(pixel[channel] for pixel in pixels) // len(pixels)
                for channel in range(3)
            ]
            phash = _dhash_from_pixels(width, height, pixels)
    return ImageInfo(width=width, height=height, mode=mode, average_color=average, perceptual_hash=phash)


def _exif_from_pillow(img) -> dict:
    try:
        from PIL.ExifTags import TAGS

        raw = img.getexif()
        exif = {}
        for key, value in raw.items():
            label = TAGS.get(key, str(key))
            if isinstance(value, bytes):
                value = value[:64].hex()
            if isinstance(value, (str, int, float, bool)) or value is None:
                exif[label] = value
            else:
                exif[label] = str(value)
        return exif
    except Exception:
        return {}


def _visual_embedding(image: ImageInfo) -> List[float]:
    values: List[float] = []
    if image.average_color:
        values.extend([channel / 255.0 for channel in image.average_color])
    else:
        values.extend([0.0, 0.0, 0.0])

    for color in image.dominant_colors[:5]:
        values.extend([channel / 255.0 for channel in color[:3]])
    while len(values) < 18:
        values.append(0.0)

    width = float(image.width or 0)
    height = float(image.height or 0)
    total = width + height
    if total:
        values.extend([width / total, height / total, min(width, height) / max(width, height)])
    else:
        values.extend([0.0, 0.0, 0.0])

    if image.perceptual_hash:
        hash_value = int(image.perceptual_hash, 16)
        for shift in range(0, 64, 8):
            values.append(((hash_value >> shift) & 0xFF) / 255.0)
    else:
        values.extend([0.0] * 8)
    return normalize(values)


def _heuristic_caption(image: ImageInfo, tags: List[str]) -> str:
    parts = []
    if image.width and image.height:
        orientation = "landscape" if image.width > image.height else "portrait" if image.height > image.width else "square"
        parts.append(f"{orientation} image")
        parts.append(f"{image.width}x{image.height}")
    elif tags:
        parts.append(f"{tags[0]} image")
    else:
        parts.append("image")
    if image.average_color:
        parts.append(f"average color rgb({image.average_color[0]}, {image.average_color[1]}, {image.average_color[2]})")
    return ", ".join(parts)


def _privacy_flags(image: ImageInfo) -> List[str]:
    flags = []
    if image.exif:
        flags.append("has_exif")
    gps_keys = {"GPSInfo", "GPSLatitude", "GPSLongitude"}
    if gps_keys.intersection(image.exif):
        flags.append("has_location_metadata")
    return flags


def _guess_media_type(path: Path, data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"P6") or data.startswith(b"P3"):
        return "image/x-portable-pixmap"
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _read_dimensions(data: bytes, media_type: str) -> Tuple[Optional[int], Optional[int]]:
    if media_type == "image/png" and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    if media_type == "image/gif" and len(data) >= 10:
        return struct.unpack("<HH", data[6:10])
    if media_type == "image/jpeg":
        return _jpeg_dimensions(data)
    if media_type == "image/webp":
        return _webp_dimensions(data)
    return None, None


def _jpeg_dimensions(data: bytes) -> Tuple[Optional[int], Optional[int]]:
    index = 2
    while index + 9 < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        index += 2
        if marker in (0xD8, 0xD9):
            continue
        length = int.from_bytes(data[index:index + 2], "big")
        if marker in range(0xC0, 0xC4) or marker in range(0xC5, 0xC8) or marker in range(0xC9, 0xCC) or marker in range(0xCD, 0xD0):
            height = int.from_bytes(data[index + 3:index + 5], "big")
            width = int.from_bytes(data[index + 5:index + 7], "big")
            return width, height
        index += length
    return None, None


def _webp_dimensions(data: bytes) -> Tuple[Optional[int], Optional[int]]:
    if len(data) < 30:
        return None, None
    chunk = data[12:16]
    if chunk == b"VP8X":
        width = 1 + int.from_bytes(data[24:27], "little")
        height = 1 + int.from_bytes(data[27:30], "little")
        return width, height
    return None, None


def _read_ppm(data: bytes) -> Tuple[Optional[int], Optional[int], List[Tuple[int, int, int]]]:
    stream = _TokenStream(data)
    magic = stream.next_token()
    if magic not in (b"P6", b"P3"):
        return None, None, []
    width = int(stream.next_token())
    height = int(stream.next_token())
    max_value = int(stream.next_token())
    if max_value <= 0 or max_value > 255:
        return width, height, []
    if magic == b"P6":
        raw = stream.remaining()
        pixels = [
            (raw[i], raw[i + 1], raw[i + 2])
            for i in range(0, min(len(raw), width * height * 3), 3)
            if i + 2 < len(raw)
        ]
    else:
        tokens = [int(token) for token in stream.all_remaining_tokens()]
        pixels = [
            (tokens[i], tokens[i + 1], tokens[i + 2])
            for i in range(0, min(len(tokens), width * height * 3), 3)
            if i + 2 < len(tokens)
        ]
    return width, height, pixels


class _TokenStream:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.index = 0

    def next_token(self) -> bytes:
        self._skip_ws_and_comments()
        start = self.index
        while self.index < len(self.data) and self.data[self.index] not in b" \t\r\n":
            self.index += 1
        return self.data[start:self.index]

    def remaining(self) -> bytes:
        if self.index < len(self.data) and self.data[self.index] in b" \t\r\n":
            self.index += 1
        return self.data[self.index:]

    def all_remaining_tokens(self) -> List[bytes]:
        tokens = []
        while True:
            self._skip_ws_and_comments()
            if self.index >= len(self.data):
                return tokens
            tokens.append(self.next_token())

    def _skip_ws_and_comments(self) -> None:
        while self.index < len(self.data):
            if self.data[self.index] in b" \t\r\n":
                self.index += 1
                continue
            if self.data[self.index:self.index + 1] == b"#":
                while self.index < len(self.data) and self.data[self.index] not in b"\r\n":
                    self.index += 1
                continue
            return


def _dhash_from_pixels(width: int, height: int, pixels: List[Tuple[int, int, int]]) -> Optional[str]:
    if not width or not height or not pixels:
        return None
    gray = [sum(pixel) // 3 for pixel in pixels]
    bits = []
    for row in range(8):
        y = min(height - 1, int(row * height / 8))
        for col in range(8):
            x1 = min(width - 1, int(col * width / 9))
            x2 = min(width - 1, int((col + 1) * width / 9))
            bits.append(1 if gray[y * width + x1] > gray[y * width + x2] else 0)
    return _bits_to_hex(bits)


def _bits_to_hex(bits: List[int]) -> str:
    value = 0
    for bit in bits:
        value = (value << 1) | bit
    return f"{value:016x}"


def _basic_tags(media_type: str, image: ImageInfo) -> List[str]:
    tags = []
    if media_type.startswith("image/"):
        tags.append(media_type.removeprefix("image/"))
    if image.width and image.height:
        tags.append("landscape" if image.width > image.height else "portrait" if image.height > image.width else "square")
    return tags
