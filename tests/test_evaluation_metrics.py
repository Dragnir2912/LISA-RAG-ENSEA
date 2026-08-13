from __future__ import annotations

import unittest

from src.evaluation_metrics import (
    answerable_automated_pass,
    is_abstention,
    reference_fact_metrics,
    route_compatible,
)


class EvaluationMetricsTests(unittest.TestCase):
    def test_decimal_and_singular_unit_are_equivalent(self) -> None:
        metrics = reference_fact_metrics(
            "11,9 minutes.",
            "Il fallait en moyenne 11.9 minute [1].",
        )
        self.assertEqual(metrics.numeric_recall, 1.0)
        self.assertEqual(metrics.unit_recall, 1.0)
        self.assertTrue(answerable_automated_pass(
            "11,9 minutes.", "Il fallait 11.9 minute [1].",
            citation_source_hit=True,
        ))

    def test_optional_estimation_word_does_not_create_a_false_failure(self) -> None:
        self.assertTrue(answerable_automated_pass(
            "94,3 % (estimation).", "Le taux est de 94,3 % [1].",
            citation_source_hit=True,
        ))

    def test_ratio_without_denominator_remains_partial(self) -> None:
        self.assertFalse(answerable_automated_pass(
            "109,3 hommes pour 100 femmes.", "Le rapport est de 109,3 [1].",
            citation_source_hit=True,
        ))

    def test_natural_abstention_variants_are_recognised(self) -> None:
        variants = (
            "Je ne trouve donc pas cette information dans les sources.",
            "Aucune donnée chiffrée spécifique pour 2025 n'est fournie.",
            "Les sources n'informent pas spécifiquement sur cet indicateur.",
            "Les tableaux historiques sont présentés sans fournir la valeur demandée.",
        )
        self.assertTrue(all(is_abstention(answer) for answer in variants))

    def test_all_is_compatible_but_not_exact(self) -> None:
        self.assertTrue(route_compatible("all", "facebook"))
        self.assertTrue(route_compatible("all", "economic"))
        self.assertFalse(route_compatible("institutional", "economic"))


if __name__ == "__main__":
    unittest.main()
