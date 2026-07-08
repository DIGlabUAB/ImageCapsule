from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

from .capsule import Capsule
from .extractors import from_file
from .store import Store


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="imgcapsule", description="Build and search portable image capsules.")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Create a capsule JSON sidecar for one image.")
    build.add_argument("image")
    build.add_argument("-o", "--out")

    inspect = sub.add_parser("inspect", help="Print a capsule JSON file.")
    inspect.add_argument("capsule")

    index = sub.add_parser("index", help="Index a folder or image into a capsule store.")
    index.add_argument("path")
    index.add_argument("--db", default="images.icdb")
    index.add_argument("--no-recursive", action="store_true")

    search = sub.add_parser("search", help="Search a capsule store.")
    search.add_argument("db")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=10)

    duplicates = sub.add_parser("duplicates", help="Find near-duplicates for an image or perceptual hash.")
    duplicates.add_argument("db")
    duplicates.add_argument("image_or_hash")
    duplicates.add_argument("--max-distance", type=int, default=8)
    duplicates.add_argument("--limit", type=int, default=10)

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "build":
        capsule = from_file(args.image)
        out = Path(args.out) if args.out else Path(f"{args.image}.capsule.json")
        capsule.save(out)
        print(out)
        return 0

    if args.command == "inspect":
        capsule = Capsule.load(args.capsule)
        print(capsule.to_json())
        return 0

    if args.command == "index":
        with Store(args.db) as store:
            path = Path(args.path)
            if path.is_dir():
                capsules = store.add_folder(path, recursive=not args.no_recursive)
                print(json.dumps({"indexed": len(capsules), "db": args.db}, indent=2))
            else:
                store.add_path(path)
                print(json.dumps({"indexed": 1, "db": args.db}, indent=2))
        return 0

    if args.command == "search":
        with Store(args.db) as store:
            print(json.dumps(store.search(args.query, limit=args.limit), indent=2))
        return 0

    if args.command == "duplicates":
        with Store(args.db) as store:
            print(json.dumps(store.near_duplicates(args.image_or_hash, max_distance=args.max_distance, limit=args.limit), indent=2))
        return 0

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
