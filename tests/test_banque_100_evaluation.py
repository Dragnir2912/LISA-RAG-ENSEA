from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pypdf import PdfReader

from src.evaluation_report import export_evaluation_pdf
from src.golden_set_word import build_banque_100_golden_set


ROOT = Path(__file__).resolve().parents[1]


class Banque100EvaluationTests(unittest.TestCase):
    def test_word_builds_a_complete_traceable_golden_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "banque_100.jsonl"
            report = build_banque_100_golden_set(
                ROOT / "data/questions/Banque_100_questions_CORRIGEE.docx",
                output_path,
                facebook_raw_dir=ROOT / "data/facebook/raw",
                facebook_chunks_path=ROOT / "data/facebook/chunks.jsonl",
            )
            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(report.total_questions, 100)
        self.assertEqual(report.answerable_questions, 75)
        self.assertEqual(report.unanswerable_questions, 25)
        self.assertEqual(report.pdf_questions, 65)
        self.assertEqual(report.facebook_questions, 10)
        self.assertEqual([row["question_number"] for row in rows], list(range(1, 101)))
        self.assertTrue(all(row["relevant_urls"] for row in rows[:70]))
        self.assertTrue(all(rows[index - 1]["relevant_urls"] for index in (72, 73, 74, 77, 99)))
        self.assertTrue(all(
            not row["relevant_urls"]
            for row in rows
            if row["question_number"] not in {*range(1, 71), 72, 73, 74, 77, 99}
        ))
        self.assertEqual(rows[0]["reference_page"], 10)
        self.assertEqual(rows[60]["expected_route"], "facebook")
        self.assertIn("statistique agricole", rows[60]["source_title"].casefold())
        self.assertEqual(rows[28]["reference_answer"], "-0,4 %.")
        self.assertEqual(rows[29]["reference_answer"], "6,3 %.")
        self.assertEqual(rows[28]["original_reference_answer"], "")
        self.assertEqual(rows[28]["reference_correction_reason"], "")
        self.assertEqual(rows[71]["reference_answer"], "12 083,3 milliards de FCFA.")
        self.assertEqual(len(rows[76]["relevant_urls"]), 2)
        self.assertEqual(rows[98]["reference_answer"], "7,93 %.")

    def test_pdf_export_contains_questions_answers_sources_and_latencies(self) -> None:
        row = {
            "question_number": 1,
            "question": "Quelle était la population de la Côte d’Ivoire en 2025 ?",
            "should_answer": True,
            "automated_pass": True,
            "domain_label": "Démographie",
            "route": "economic",
            "expected_route": "economic",
            "total_seconds": 7.52,
            "retrieval_seconds": 0.41,
            "generation_seconds": 7.03,
            "reference_answer": "32 757 857 habitants.",
            "generated_answer": "La population était de 32 757 857 habitants [1].",
            "source_locator": "Chiffres_Cles_CIV_2025.pdf - page 10",
            "citations": [{
                "title": "Chiffres clés CIV 2025",
                "section": "Page 10",
                "url": "/api/source-files/ANSTAT/Chiffres_Cles_CIV_2025.pdf",
            }],
        }
        summary = {
            "questions_total": 1,
            "questions_completed": 1,
            "errors": 0,
            "answerable_questions": 1,
            "unanswerable_questions": 0,
            "automated_pass_rate": 1.0,
            "mean_total_seconds": 7.52,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "evaluation.pdf"
            export_evaluation_pdf([row], summary, output_path)
            reader = PdfReader(output_path)
            extracted = "\n".join(page.extract_text() or "" for page in reader.pages)

        self.assertIn("32 757 857", extracted)
        self.assertIn("7.52", extracted)
        self.assertIn("Chiffres clés", extracted)


if __name__ == "__main__":
    unittest.main()
