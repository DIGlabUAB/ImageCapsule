from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

from .capsule import Capsule
from .enrich import enrich_from_json_file
from .extractors import from_file
from .store import Store


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="imgcapsule", description="Build and search portable image capsules.")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Create a capsule JSON sidecar for one image.")
    build.add_argument("image")
    build.add_argument("-o", "--out")
    build.add_argument("--adapter", action="append", default=[], help="Optional extractor adapter, e.g. ocr.")

    inspect = sub.add_parser("inspect", help="Print a capsule JSON file.")
    inspect.add_argument("capsule")

    validate = sub.add_parser("validate", help="Validate a capsule JSON file.")
    validate.add_argument("capsule")

    enrich = sub.add_parser("enrich", help="Merge external model output into a capsule.")
    enrich.add_argument("capsule")
    enrich.add_argument("json")
    enrich.add_argument("-o", "--out")

    index = sub.add_parser("index", help="Index a folder or image into a capsule store.")
    index.add_argument("path")
    index.add_argument("--db", default="images.icdb")
    index.add_argument("--no-recursive", action="store_true")
    index.add_argument("--adapter", action="append", default=[], help="Optional extractor adapter, e.g. ocr.")

    list_cmd = sub.add_parser("list", help="List recently indexed capsules.")
    list_cmd.add_argument("db")
    list_cmd.add_argument("--limit", type=int, default=50)

    search = sub.add_parser("search", help="Search a capsule store.")
    search.add_argument("db")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=10)

    duplicates = sub.add_parser("duplicates", help="Find near-duplicates for an image or perceptual hash.")
    duplicates.add_argument("db")
    duplicates.add_argument("image_or_hash")
    duplicates.add_argument("--max-distance", type=int, default=8)
    duplicates.add_argument("--limit", type=int, default=10)

    similar = sub.add_parser("similar", help="Find visually similar images using stored embeddings.")
    similar.add_argument("db")
    similar.add_argument("image_or_sha")
    similar.add_argument("--limit", type=int, default=10)

    refresh = sub.add_parser("refresh", help="Refresh indexed capsules from their source images.")
    refresh.add_argument("db")
    refresh.add_argument("--adapter", action="append", default=[], help="Optional extractor adapter, e.g. ocr.")

    export = sub.add_parser("export", help="Export a store as JSON Lines.")
    export.add_argument("db")
    export.add_argument("out")

    import_cmd = sub.add_parser("import", help="Import JSON Lines capsules into a store.")
    import_cmd.add_argument("db")
    import_cmd.add_argument("jsonl")

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "build":
        capsule = from_file(args.image, adapters=args.adapter)
        out = Path(args.out) if args.out else Path(f"{args.image}.capsule.json")
        capsule.save(out)
        print(out)
        return 0

    if args.command == "inspect":
        capsule = Capsule.load(args.capsule)
        print(capsule.to_json())
        return 0

    if args.command == "validate":
        capsule = Capsule.load(args.capsule)
        errors = capsule.validate()
        if errors:
            print(json.dumps({"ok": False, "errors": errors}, indent=2))
            return 1
        print(json.dumps({"ok": True}, indent=2))
        return 0

    if args.command == "enrich":
        capsule = Capsule.load(args.capsule)
        enrich_from_json_file(capsule, args.json)
        out = Path(args.out) if args.out else Path(args.capsule)
        capsule.save(out)
        print(out)
        return 0

    if args.command == "index":
        with Store(args.db) as store:
            path = Path(args.path)
            if path.is_dir():
                capsules = store.add_folder(path, recursive=not args.no_recursive, adapters=args.adapter)
                print(json.dumps({"indexed": len(capsules), "db": args.db}, indent=2))
            else:
                store.add_path(path, adapters=args.adapter)
                print(json.dumps({"indexed": 1, "db": args.db}, indent=2))
        return 0

    if args.command == "list":
        with Store(args.db) as store:
            print(json.dumps(store.list(limit=args.limit), indent=2))
        return 0

    if args.command == "search":
        with Store(args.db) as store:
            print(json.dumps(store.search(args.query, limit=args.limit), indent=2))
        return 0

    if args.command == "duplicates":
        with Store(args.db) as store:
            print(json.dumps(store.near_duplicates(args.image_or_hash, max_distance=args.max_distance, limit=args.limit), indent=2))
        return 0

    if args.command == "similar":
        with Store(args.db) as store:
            print(json.dumps(store.similar(args.image_or_sha, limit=args.limit), indent=2))
        return 0

    if args.command == "refresh":
        with Store(args.db) as store:
            count = store.refresh(adapters=args.adapter)
            print(json.dumps({"refreshed": count, "db": args.db}, indent=2))
        return 0

    if args.command == "export":
        with Store(args.db) as store:
            out = store.export_jsonl(args.out)
            print(out)
        return 0

    if args.command == "import":
        with Store(args.db) as store:
            count = store.import_jsonl(args.jsonl)
            print(json.dumps({"imported": count, "db": args.db}, indent=2))
        return 0

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
