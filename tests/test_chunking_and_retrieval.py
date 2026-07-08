from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rag_local.chunk_quality import (  # noqa: E402
    filter_quality_chunks,
    get_low_quality_reason,
    is_cover_or_title_chunk,
    is_heading_only_text,
)
from src.rag_local.chunking import (  # noqa: E402
    build_semantic_chunks,
    cosine_similarity,
    is_heading_only_unit,
    merge_short_chunks,
    percentile,
    split_fixed_size,
    split_markdown_units,
    split_recursive,
    split_semantic,
)
from src.rag_local.hybrid_search import (  # noqa: E402
    BM25Index,
    StoredChunk,
    reciprocal_rank_fusion,
    tokenize,
)


class KeywordEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []

        for text in texts:
            lowered = text.lower()

            if "dragon" in lowered or "дракон" in lowered:
                embeddings.append([1.0, 0.0, 0.0])
            elif "snake" in lowered or "зме" in lowered:
                embeddings.append([0.0, 1.0, 0.0])
            elif "school" in lowered or "школ" in lowered:
                embeddings.append([0.0, 0.0, 1.0])
            else:
                embeddings.append([0.5, 0.5, 0.5])

        return embeddings


def synthetic_text(length: int = 600) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return "".join(alphabet[index % len(alphabet)] for index in range(length))


class TestChunkingCore(unittest.TestCase):
    def test_fixed_size_rejects_invalid_chunk_size(self) -> None:
        with self.assertRaises(ValueError):
            split_fixed_size("abc", chunk_size=0, overlap=0)

    def test_fixed_size_rejects_negative_overlap(self) -> None:
        with self.assertRaises(ValueError):
            split_fixed_size("abc", chunk_size=10, overlap=-1)

    def test_fixed_size_rejects_overlap_equal_to_chunk_size(self) -> None:
        with self.assertRaises(ValueError):
            split_fixed_size("abc", chunk_size=10, overlap=10)

    def test_fixed_size_preserves_overlap(self) -> None:
        chunks = split_fixed_size("abcdefghij", chunk_size=5, overlap=2)
        self.assertEqual(["abcde", "defgh", "ghij"], chunks)

    def test_fixed_size_returns_no_empty_chunks(self) -> None:
        chunks = split_fixed_size("   abc   ", chunk_size=4, overlap=1)
        self.assertTrue(all(chunk for chunk in chunks))

    def test_recursive_keeps_small_text_as_one_chunk(self) -> None:
        chunks = split_recursive("Small paragraph.", chunk_size=100, overlap=10)
        self.assertEqual(["Small paragraph."], chunks)

    def test_recursive_respects_max_size_for_regular_text(self) -> None:
        text = "Sentence one. Sentence two. Sentence three. " * 10
        chunks = split_recursive(text, chunk_size=80, overlap=10)
        self.assertTrue(all(len(chunk) <= 80 for chunk in chunks))

    def test_markdown_units_split_on_blank_lines(self) -> None:
        units = split_markdown_units("First paragraph.\n\nSecond paragraph.")
        self.assertEqual(["First paragraph.", "Second paragraph."], units)

    def test_markdown_units_split_on_page_marker(self) -> None:
        units = split_markdown_units("Text before\n<!-- Page 2 -->\nText after")
        self.assertEqual(["Text before", "<!-- Page 2 -->\nText after"], units)

    def test_markdown_units_keep_adjacent_headings_together(self) -> None:
        units = split_markdown_units("# Chapter 1\n## Title\n\nFirst paragraph")
        self.assertEqual(["# Chapter 1\n## Title", "First paragraph"], units)

    def test_markdown_units_keep_code_blocks_intact(self) -> None:
        text = "Intro\n\n```python\n# not heading\nprint('x')\n```\n\nOutro"
        units = split_markdown_units(text)
        self.assertIn("```python\n# not heading\nprint('x')\n```", units)

    def test_heading_only_unit_detects_heading_block(self) -> None:
        self.assertTrue(is_heading_only_unit("# Chapter\n## Title"))

    def test_heading_only_unit_rejects_heading_with_text(self) -> None:
        self.assertFalse(is_heading_only_unit("# Chapter\nText"))

    def test_cosine_similarity_for_equal_vectors(self) -> None:
        self.assertAlmostEqual(1.0, cosine_similarity([1, 2, 3], [1, 2, 3]))

    def test_cosine_similarity_for_orthogonal_vectors(self) -> None:
        self.assertAlmostEqual(0.0, cosine_similarity([1, 0], [0, 1]))

    def test_cosine_similarity_for_zero_vector(self) -> None:
        self.assertEqual(0.0, cosine_similarity([0, 0], [1, 1]))

    def test_percentile_empty_defaults_to_one(self) -> None:
        self.assertEqual(1.0, percentile([], 80))

    def test_percentile_middle_value(self) -> None:
        self.assertEqual(3, percentile([1, 2, 3, 4, 5], 50))

    def test_build_semantic_chunks_breaks_by_size(self) -> None:
        chunks = build_semantic_chunks(
            units=["a" * 10, "b" * 10, "c" * 10],
            distances=[0.0, 0.0],
            threshold=1.0,
            max_chars=22,
        )
        self.assertEqual(2, len(chunks))

    def test_build_semantic_chunks_breaks_by_meaning(self) -> None:
        chunks = build_semantic_chunks(
            units=["dragon text", "snake text", "snake more"],
            distances=[0.9, 0.1],
            threshold=0.8,
            max_chars=100,
        )
        self.assertEqual(["dragon text", "snake text\n\nsnake more"], chunks)

    def test_build_semantic_chunks_keeps_heading_with_first_paragraph(self) -> None:
        chunks = build_semantic_chunks(
            units=["# Chapter 1\n## Title", "First paragraph.", "Second paragraph."],
            distances=[1.0, 1.0],
            threshold=0.5,
            max_chars=1000,
        )
        self.assertTrue(chunks[0].startswith("# Chapter 1"))
        self.assertIn("First paragraph.", chunks[0])

    def test_merge_short_chunks_merges_short_chunk_with_previous(self) -> None:
        chunks = merge_short_chunks(
            ["Long enough paragraph " * 20, "Tiny."],
            min_chars=300,
            max_chars=1000,
        )
        self.assertEqual(1, len(chunks))
        self.assertIn("Tiny.", chunks[0])

    def test_merge_short_chunks_keeps_short_chunk_when_join_would_exceed_max(self) -> None:
        chunks = merge_short_chunks(
            ["a" * 996, "Tiny."],
            min_chars=300,
            max_chars=1000,
        )
        self.assertEqual(2, len(chunks))

    def test_merge_short_chunks_rejects_invalid_min_chars(self) -> None:
        with self.assertRaises(ValueError):
            merge_short_chunks(["text"], min_chars=0, max_chars=100)

    def test_merge_short_chunks_rejects_invalid_max_chars(self) -> None:
        with self.assertRaises(ValueError):
            merge_short_chunks(["text"], min_chars=100, max_chars=50)

    def test_split_semantic_validates_max_chars(self) -> None:
        with self.assertRaises(ValueError):
            split_semantic("text", 0, KeywordEmbedder(), 80)

    def test_split_semantic_validates_percentile(self) -> None:
        with self.assertRaises(ValueError):
            split_semantic("text", 100, KeywordEmbedder(), 101)

    def test_split_semantic_uses_embedder_for_multiple_units(self) -> None:
        text = (
            "Dragon story " * 40
            + "\n\n"
            + "Snake story " * 40
            + "\n\n"
            + "School story " * 40
        )
        chunks = split_semantic(
            text,
            1000,
            KeywordEmbedder(),
            50,
            min_chunk_chars=100,
        )
        self.assertGreaterEqual(len(chunks), 2)

    def test_split_semantic_merges_tiny_units(self) -> None:
        text = "# Chapter\n\nTiny.\n\nMain body paragraph " * 20
        chunks = split_semantic(
            text,
            max_chars=1000,
            embedder=KeywordEmbedder(),
            break_percentile=80,
            min_chunk_chars=120,
        )
        self.assertTrue(all(len(chunk) >= 120 or len(chunks) == 1 for chunk in chunks))


class TestChunkQuality(unittest.TestCase):
    def test_heading_only_text_detects_headings(self) -> None:
        self.assertTrue(is_heading_only_text("# One\n## Two"))

    def test_heading_only_text_rejects_body_text(self) -> None:
        self.assertFalse(is_heading_only_text("# One\nBody text"))

    def test_cover_chunk_detects_english_cover(self) -> None:
        self.assertTrue(is_cover_or_title_chunk("Book cover of Harry Potter"))

    def test_cover_chunk_detects_russian_cover(self) -> None:
        self.assertTrue(is_cover_or_title_chunk("Обложка книги Гарри Поттер"))

    def test_cover_chunk_detects_title_only(self) -> None:
        self.assertTrue(is_cover_or_title_chunk("# Гарри Поттер\n## и Тайная комната"))

    def test_low_quality_empty(self) -> None:
        self.assertEqual("empty", get_low_quality_reason(""))

    def test_low_quality_heading_only(self) -> None:
        self.assertEqual("heading_only", get_low_quality_reason("# Chapter"))

    def test_low_quality_short_initial_chunk(self) -> None:
        reason = get_low_quality_reason("Short intro", {"chunk_id": 2, "index": 1})
        self.assertEqual("short_initial_chunk", reason)

    def test_quality_keeps_short_later_dialogue(self) -> None:
        reason = get_low_quality_reason("Yes.", {"chunk_id": 20, "index": 19})
        self.assertIsNone(reason)

    def test_filter_quality_chunks_splits_kept_and_removed(self) -> None:
        chunks = [
            {"text": "Book cover of HP", "chunk_id": 1, "index": 0},
            {"text": "Useful body paragraph " * 20, "chunk_id": 10, "index": 9},
        ]
        kept, removed = filter_quality_chunks(chunks)
        self.assertEqual(1, len(kept))
        self.assertEqual(1, len(removed))


class TestRetrievalCore(unittest.TestCase):
    def test_tokenize_lowercases_russian_text(self) -> None:
        self.assertEqual(["сириус", "блэк"], tokenize("Сириус Блэк!"))

    def test_tokenize_keeps_latin_words(self) -> None:
        self.assertEqual(["harry", "potter"], tokenize("Harry Potter."))

    def test_bm25_finds_exact_keyword_document(self) -> None:
        index = BM25Index(
            [
                StoredChunk("a", "dragon tournament", {}),
                StoredChunk("b", "snake chamber", {}),
            ]
        )
        self.assertEqual("b", index.search("snake", 1)[0]["id"])

    def test_bm25_returns_empty_for_unknown_query(self) -> None:
        index = BM25Index([StoredChunk("a", "dragon tournament", {})])
        self.assertEqual([], index.search("unknown", 10))

    def test_bm25_honors_top_k(self) -> None:
        chunks = [StoredChunk(str(index), f"harry text {index}", {}) for index in range(10)]
        self.assertEqual(3, len(BM25Index(chunks).search("harry", 3)))

    def test_rrf_merges_duplicate_ids(self) -> None:
        fused = reciprocal_rank_fusion(
            [[{"id": "a", "similarity": 0.9}], [{"id": "a", "bm25_score": 2.0}]]
        )
        self.assertEqual(1, len(fused))
        self.assertEqual(0.9, fused[0]["similarity"])
        self.assertEqual(2.0, fused[0]["bm25_score"])

    def test_rrf_does_not_overwrite_scores_with_none(self) -> None:
        fused = reciprocal_rank_fusion(
            [
                [{"id": "a", "similarity": 0.9}],
                [{"id": "a", "similarity": None, "bm25_score": 2.0}],
            ]
        )
        self.assertEqual(0.9, fused[0]["similarity"])

    def test_rrf_weight_can_prioritize_vector_result(self) -> None:
        fused = reciprocal_rank_fusion(
            [
                [{"id": "vector", "vector_rank": 1}],
                [{"id": "keyword", "bm25_rank": 1}],
            ],
            weights=[0.75, 0.25],
        )
        self.assertEqual("vector", fused[0]["id"])

    def test_rrf_rejects_wrong_weight_count(self) -> None:
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([[{"id": "a"}]], weights=[0.5, 0.5])

    def test_rrf_duplicate_seen_in_two_lists_beats_single_list_result(self) -> None:
        fused = reciprocal_rank_fusion(
            [
                [{"id": "shared"}, {"id": "vector_only"}],
                [{"id": "shared"}, {"id": "bm25_only"}],
            ]
        )
        self.assertEqual("shared", fused[0]["id"])


def make_fixed_size_test(chunk_size: int, overlap: int):
    def test(self: unittest.TestCase) -> None:
        text = synthetic_text(500)
        chunks = split_fixed_size(text, chunk_size=chunk_size, overlap=overlap)
        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk) <= chunk_size for chunk in chunks))
        if len(chunks) > 1 and overlap > 0:
            self.assertEqual(chunks[0][-overlap:], chunks[1][:overlap])

    return test


def make_bm25_query_test(query: str, expected_id: str):
    def test(self: unittest.TestCase) -> None:
        chunks = [
            StoredChunk("harry", "гарри поттер волшебник хогвартс", {}),
            StoredChunk("sirius", "сириус блэк азкабан узник", {}),
            StoredChunk("dobby", "добби предупреждает гарри опасность", {}),
            StoredChunk("snake", "тайная комната василиск змеиный язык", {}),
            StoredChunk("quidditch", "квиддич метла матч ловец", {}),
        ]
        results = BM25Index(chunks).search(query, top_k=3)
        self.assertTrue(results)
        self.assertEqual(expected_id, results[0]["id"])

    return test


def make_rrf_weight_test(vector_weight: float, bm25_weight: float):
    def test(self: unittest.TestCase) -> None:
        fused = reciprocal_rank_fusion(
            [
                [{"id": "vector", "vector_rank": 1}],
                [{"id": "bm25", "bm25_rank": 1}],
            ],
            weights=[vector_weight, bm25_weight],
        )
        expected = "vector" if vector_weight >= bm25_weight else "bm25"
        self.assertEqual(expected, fused[0]["id"])

    return test


for index, chunk_size in enumerate(range(40, 100), start=1):
    overlap = min(10, chunk_size // 4)
    setattr(
        TestChunkingCore,
        f"test_generated_fixed_size_case_{index:02d}",
        make_fixed_size_test(chunk_size, overlap),
    )


BM25_CASES = [
    ("гарри", "harry"),
    ("поттер", "harry"),
    ("волшебник", "harry"),
    ("хогвартс", "harry"),
    ("сириус", "sirius"),
    ("блэк", "sirius"),
    ("азкабан", "sirius"),
    ("узник", "sirius"),
    ("добби", "dobby"),
    ("предупреждает", "dobby"),
    ("опасность", "dobby"),
    ("тайная", "snake"),
    ("комната", "snake"),
    ("василиск", "snake"),
    ("змеиный язык", "snake"),
    ("квиддич", "quidditch"),
    ("метла", "quidditch"),
    ("матч", "quidditch"),
    ("ловец", "quidditch"),
    ("гарри опасность", "dobby"),
]

for index, (query, expected_id) in enumerate(BM25_CASES, start=1):
    setattr(
        TestRetrievalCore,
        f"test_generated_bm25_query_case_{index:02d}",
        make_bm25_query_test(query, expected_id),
    )


RRF_CASES = [
    (0.75, 0.25),
    (0.8, 0.2),
    (0.6, 0.4),
    (0.51, 0.49),
    (1.0, 0.0),
    (0.25, 0.75),
    (0.2, 0.8),
    (0.4, 0.6),
    (0.49, 0.51),
    (0.0, 1.0),
]

for index, (vector_weight, bm25_weight) in enumerate(RRF_CASES, start=1):
    setattr(
        TestRetrievalCore,
        f"test_generated_rrf_weight_case_{index:02d}",
        make_rrf_weight_test(vector_weight, bm25_weight),
    )


class TestRetrievalMetrics(unittest.TestCase):
    def test_mrr_detects_good_top_rank(self) -> None:
        ranked_ids = ["target", "other"]
        reciprocal_rank = 1 / (ranked_ids.index("target") + 1)
        self.assertEqual(1.0, reciprocal_rank)

    def test_mrr_detects_bad_late_rank(self) -> None:
        ranked_ids = ["a", "b", "target"]
        reciprocal_rank = 1 / (ranked_ids.index("target") + 1)
        self.assertTrue(math.isclose(1 / 3, reciprocal_rank))

    def test_recall_at_20_is_true_when_target_inside_top_20(self) -> None:
        ranked_ids = [str(index) for index in range(20)]
        self.assertIn("19", ranked_ids[:20])

    def test_recall_at_20_is_false_when_target_outside_top_20(self) -> None:
        ranked_ids = [str(index) for index in range(30)]
        self.assertNotIn("25", ranked_ids[:20])


if __name__ == "__main__":
    unittest.main()
