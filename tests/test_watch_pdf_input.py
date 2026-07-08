from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.watch_pdf_input import (  # noqa: E402
    build_document_metadata,
    build_extract_command,
    build_file_signature,
    build_ingest_command,
    mark_state,
    should_process_pdf,
    slugify,
)


def make_args(**overrides: object) -> argparse.Namespace:
    values = {
        "default_author": "Unknown",
        "default_year": 2026,
        "document_type": "book",
        "tier": "cost_effective",
        "collection": None,
        "semantic_max_chars": 2000,
        "semantic_min_chars": 300,
        "break_percentile": 80.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class TestWatchPdfInput(unittest.TestCase):
    def test_slugify_keeps_safe_document_id(self) -> None:
        self.assertEqual("harry_potter_4", slugify("Harry Potter 4!"))

    def test_build_document_metadata_uses_sidecar_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "book.pdf"
            pdf_path.write_bytes(b"%PDF")
            pdf_path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "doc_id": "hp_4",
                        "title": "Harry Potter and the Goblet of Fire",
                        "author": "J. K. Rowling",
                        "year": 2000,
                        "document_type": "book",
                    }
                ),
                encoding="utf-8",
            )

            metadata = build_document_metadata(pdf_path, make_args())

        self.assertEqual("hp_4", metadata["doc_id"])
        self.assertEqual("Harry Potter and the Goblet of Fire", metadata["title"])
        self.assertEqual("J. K. Rowling", metadata["author"])
        self.assertEqual(2000, metadata["year"])

    def test_build_document_metadata_uses_defaults_without_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "My_Document.pdf"
            pdf_path.write_bytes(b"%PDF")

            metadata = build_document_metadata(
                pdf_path,
                make_args(default_author="Default Author", default_year=1999),
            )

        self.assertEqual("my_document", metadata["doc_id"])
        self.assertEqual("My Document", metadata["title"])
        self.assertEqual("Default Author", metadata["author"])
        self.assertEqual(1999, metadata["year"])

    def test_should_process_new_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "new.pdf"
            pdf_path.write_bytes(b"%PDF")

            self.assertTrue(should_process_pdf(pdf_path, {"records": {}}, False))

    def test_should_skip_done_pdf_with_same_signature(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "done.pdf"
            pdf_path.write_bytes(b"%PDF")
            state = {"records": {}}
            mark_state(state, pdf_path, "done")

            self.assertFalse(should_process_pdf(pdf_path, state, False))

    def test_should_reprocess_pdf_when_sidecar_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "book.pdf"
            pdf_path.write_bytes(b"%PDF")
            state = {"records": {}}
            mark_state(state, pdf_path, "done")
            pdf_path.with_suffix(".json").write_text('{"year": 2000}', encoding="utf-8")

            self.assertNotEqual(
                state["records"][str(pdf_path.resolve())]["signature"],
                build_file_signature(pdf_path),
            )
            self.assertTrue(should_process_pdf(pdf_path, state, False))

    def test_build_extract_command_uses_selected_tier(self) -> None:
        command = build_extract_command(
            Path("input.pdf"),
            Path("output.md"),
            tier="agentic",
        )

        self.assertIn("input.pdf", command)
        self.assertIn("output.md", command)
        self.assertIn("agentic", command)

    def test_build_ingest_command_contains_metadata(self) -> None:
        command = build_ingest_command(
            Path("output.md"),
            {
                "doc_id": "hp_1",
                "title": "Harry Potter",
                "author": "J. K. Rowling",
                "year": 1997,
                "document_type": "book",
            },
            make_args(collection="test_collection"),
        )

        self.assertIn("--doc-id", command)
        self.assertIn("hp_1", command)
        self.assertIn("--collection", command)
        self.assertIn("test_collection", command)


if __name__ == "__main__":
    unittest.main()
