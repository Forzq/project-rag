from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rag_local.chunk_quality import get_low_quality_reason
from src.rag_local.config import DEFAULT_CHROMA_COLLECTION, DEFAULT_CHROMA_DB_PATH
from src.rag_local.vector_store import ChromaStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove low-quality cover/title chunks from a Chroma collection."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_CHROMA_DB_PATH,
        help="Directory with persistent ChromaDB data.",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_CHROMA_COLLECTION,
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--min-quality-chars",
        type=int,
        default=150,
        help="Minimum characters for short technical chunks to be kept.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete matched chunks. Without this flag the script only previews.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collection = ChromaStore(args.db_path).get_collection(args.collection)
    raw_items = collection.get(include=["documents", "metadatas"])

    ids = raw_items.get("ids", [])
    documents = raw_items.get("documents", [])
    metadatas = raw_items.get("metadatas", [])
    ids_to_delete: list[str] = []
    reasons: Counter[str] = Counter()

    for index, item_id in enumerate(ids):
        document = documents[index] or ""
        metadata = metadatas[index] or {}
        reason = get_low_quality_reason(
            document,
            metadata,
            min_characters=args.min_quality_chars,
        )

        if reason is None:
            continue

        ids_to_delete.append(str(item_id))
        reasons[reason] += 1

    print(f"Collection: {args.collection}")
    print(f"Total chunks before cleanup: {collection.count()}")
    print(f"Matched low-quality chunks: {len(ids_to_delete)}")

    for reason, count in reasons.most_common():
        print(f"- {reason}: {count}")

    if not args.apply:
        print("Dry run only. Add --apply to delete these chunks.")
        return

    if ids_to_delete:
        collection.delete(ids=ids_to_delete)

    print(f"Deleted chunks: {len(ids_to_delete)}")
    print(f"Total chunks after cleanup: {collection.count()}")


if __name__ == "__main__":
    main()
