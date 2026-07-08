from __future__ import annotations

from pathlib import Path
import json
import sqlite3
from typing import Any, Dict, Iterable, List, Optional

from .capsule import Capsule
from .extractors import from_file


class Store:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _init_db(self) -> None:
        self.conn.executescript(
            """
            create table if not exists capsules (
                id integer primary key,
                source_path text not null,
                sha256 text not null,
                media_type text not null,
                width integer,
                height integer,
                perceptual_hash text,
                caption text,
                tags text not null,
                ocr_text text,
                capsule_json text not null,
                created_at text not null,
                unique(sha256)
            );
            create index if not exists idx_capsules_path on capsules(source_path);
            create index if not exists idx_capsules_phash on capsules(perceptual_hash);
            """
        )
        try:
            self.conn.execute(
                "create virtual table if not exists capsule_fts using fts5(source_path, caption, tags, ocr_text)"
            )
        except sqlite3.OperationalError:
            self.conn.execute(
                """
                create table if not exists capsule_fts (
                    rowid integer primary key,
                    source_path text,
                    caption text,
                    tags text,
                    ocr_text text
                )
                """
            )
        self.conn.commit()

    def add_path(self, path: str | Path) -> Capsule:
        capsule = from_file(path)
        self.add_capsule(capsule)
        return capsule

    def add_folder(self, folder: str | Path, *, recursive: bool = True) -> List[Capsule]:
        root = Path(folder)
        pattern = "**/*" if recursive else "*"
        capsules = []
        for path in root.glob(pattern):
            if path.is_file() and _looks_like_image(path):
                capsules.append(self.add_path(path))
        return capsules

    def add_capsule(self, capsule: Capsule) -> None:
        payload = capsule.to_json(indent=None)
        tags = " ".join(capsule.semantic.tags)
        with self.conn:
            cursor = self.conn.execute(
                """
                insert into capsules (
                    source_path, sha256, media_type, width, height, perceptual_hash,
                    caption, tags, ocr_text, capsule_json, created_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(sha256) do update set
                    source_path=excluded.source_path,
                    media_type=excluded.media_type,
                    width=excluded.width,
                    height=excluded.height,
                    perceptual_hash=excluded.perceptual_hash,
                    caption=excluded.caption,
                    tags=excluded.tags,
                    ocr_text=excluded.ocr_text,
                    capsule_json=excluded.capsule_json
                """,
                (
                    capsule.source.path,
                    capsule.source.sha256,
                    capsule.source.media_type,
                    capsule.image.width,
                    capsule.image.height,
                    capsule.image.perceptual_hash,
                    capsule.semantic.caption,
                    tags,
                    capsule.semantic.ocr_text,
                    payload,
                    capsule.created_at,
                ),
            )
            row_id = cursor.lastrowid or self._row_id_for_sha(capsule.source.sha256)
            self.conn.execute("delete from capsule_fts where rowid = ?", (row_id,))
            self.conn.execute(
                "insert into capsule_fts(rowid, source_path, caption, tags, ocr_text) values (?, ?, ?, ?, ?)",
                (row_id, capsule.source.path, capsule.semantic.caption or "", tags, capsule.semantic.ocr_text or ""),
            )

    def get_by_sha256(self, sha256: str) -> Optional[Capsule]:
        row = self.conn.execute("select capsule_json from capsules where sha256 = ?", (sha256,)).fetchone()
        if not row:
            return None
        return Capsule.from_dict(json.loads(row["capsule_json"]))

    def search(self, query: str, *, limit: int = 10) -> List[Dict[str, Any]]:
        try:
            rows = self.conn.execute(
                """
                select c.source_path, c.sha256, c.media_type, c.width, c.height, c.tags, c.caption, c.ocr_text
                from capsule_fts f
                join capsules c on c.id = f.rowid
                where capsule_fts match ?
                limit ?
                """,
                (query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            like = f"%{query}%"
            rows = self.conn.execute(
                """
                select source_path, sha256, media_type, width, height, tags, caption, ocr_text
                from capsules
                where source_path like ? or tags like ? or caption like ? or ocr_text like ?
                limit ?
                """,
                (like, like, like, like, limit),
            ).fetchall()
        return [_row_to_result(row) for row in rows]

    def near_duplicates(self, path_or_hash: str | Path, *, max_distance: int = 8, limit: int = 10) -> List[Dict[str, Any]]:
        query_hash = str(path_or_hash)
        if Path(query_hash).exists():
            query_hash = from_file(query_hash).image.perceptual_hash or ""
        if not query_hash:
            return []
        rows = self.conn.execute(
            """
            select source_path, sha256, media_type, width, height, tags, caption, ocr_text, perceptual_hash
            from capsules
            where perceptual_hash is not null
            """
        ).fetchall()
        scored = []
        for row in rows:
            distance = hamming_hex(query_hash, row["perceptual_hash"])
            if distance <= max_distance:
                item = _row_to_result(row)
                item["distance"] = distance
                scored.append(item)
        return sorted(scored, key=lambda item: item["distance"])[:limit]

    def _row_id_for_sha(self, sha256: str) -> int:
        row = self.conn.execute("select id from capsules where sha256 = ?", (sha256,)).fetchone()
        if not row:
            raise KeyError(sha256)
        return int(row["id"])


def hamming_hex(left: str, right: str) -> int:
    return bin(int(left, 16) ^ int(right, 16)).count("1")


def _row_to_result(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "path": row["source_path"],
        "sha256": row["sha256"],
        "media_type": row["media_type"],
        "width": row["width"],
        "height": row["height"],
        "tags": row["tags"].split() if row["tags"] else [],
        "caption": row["caption"],
        "ocr_text": row["ocr_text"],
    }


def _looks_like_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".ppm", ".pnm"}
