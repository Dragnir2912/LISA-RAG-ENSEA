from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pdfplumber

from src.evaluation import (
    is_documentary_abstention,
    lead_reference_relation_score,
    reference_relation_score,
)
from src.generator import PromptBuilder
from src.local_documents import LocalPDFImporter
from src.retriever import (
    _comparison_relation_bonus,
    _derived_ratio_result,
    _is_comparative_numeric_request,
    _table_fact_bonus,
)
from src.chunker import Chunk, HierarchicalChunker
from src.table_facts import (
    extract_layout_table_facts,
    extract_matrix_table_facts,
    infer_table_context,
    is_plausible_table_fact,
    is_plausible_table_row,
    is_temporal_header_row,
    rotated_words_to_layout,
)
from src.vectorstore import BM25Store, period_keys, structured_measure_alignment


ROOT = Path(__file__).resolve().parents[1]


def _word(text: str, x0: float, top: float) -> dict:
    return {
        "text": text,
        "x0": x0,
        "x1": x0 + max(8.0, len(text) * 5.0),
        "top": top,
        "bottom": top + 8.0,
        "upright": True,
    }


class StructuredTableRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        matches = list(
            (ROOT / "data").rglob("Chiffres_Cles_CIV_2025.pdf")
        )
        if not matches:
            raise unittest.SkipTest("Le PDF Chiffres_Cles_CIV_2025 est absent.")
        with pdfplumber.open(matches[0]) as document:
            cls.page_10_facts = extract_layout_table_facts(
                document.pages[9].extract_words(), page_number=10,
            )

    def test_population_men_and_masculinity_are_exact_atomic_facts(self) -> None:
        facts = {
            (fact["indicator"].casefold(), fact["period"]): fact
            for fact in self.page_10_facts
        }
        men = facts[("population homme", "2025")]
        masculinity = facts[("rapport de masculinité", "2025")]

        self.assertEqual(men["value"], "17 103 895")
        self.assertEqual(men["normalized_value"], "17103895")
        self.assertEqual(men["page"], 10)
        self.assertEqual(masculinity["value"], "109,3")
        self.assertEqual(masculinity["normalized_value"], "109.3")

    def test_rotated_quarterly_table_recovers_exact_page_facets(self) -> None:
        matches = list((ROOT / "data").rglob(
            "Comptes_nationaux_trimestriels (1er trimestriels 2026).pdf"
        ))
        if not matches:
            self.skipTest("Le PDF des comptes trimestriels est absent.")
        with pdfplumber.open(matches[0]) as document:
            page = document.pages[13]
            rotated_words = page.extract_words(
                line_dir_rotated="ltr", char_dir_rotated="btt",
            )
            words = rotated_words_to_layout(
                rotated_words,
                page_width=float(page.width),
                page_height=float(page.height),
            )
        context = infer_table_context(
            " ".join(str(word.get("text", "")) for word in words),
            source_hint=str(matches[0]),
        )
        facts = extract_layout_table_facts(
            words,
            page_number=14,
            default_unit=context["unit"],
            territory=context["territory"],
            measure_type=context["measure_type"],
        )
        pib_t4 = next(
            fact for fact in facts
            if fact["indicator"].upper() == "PIB"
            and "quarter:4:2025" in period_keys(fact["period"])
        )
        self.assertEqual(pib_t4["value"], "12 083,3")
        self.assertEqual(pib_t4["page"], 14)
        self.assertEqual(pib_t4["territory"], "Cote d'Ivoire")
        self.assertIn("milliards", pib_t4["unit"])
        self.assertIn("CVS", pib_t4["measure_type"])
        self.assertIn("volume", pib_t4["measure_type"])

    def test_spreadsheet_tables_are_atomic_and_measure_aware(self) -> None:
        matches = list((ROOT / "data").rglob(
            "COMPTES_NATIONAUX_ANNUELS_DE_2015_A_2023_vf.xlsx"
        ))
        if not matches:
            self.skipTest("Le classeur des comptes annuels est absent.")
        importer = object.__new__(LocalPDFImporter)
        _text, facts, sheets, failures = importer._extract_spreadsheet(matches[0])
        self.assertEqual(sheets, 1)
        self.assertEqual(failures, 0)
        nominal_pib = next(
            fact for fact in facts
            if fact["indicator"].casefold() == "pib"
            and "2023" in fact["period"]
            and fact["measure_type"] == "nominal"
        )
        real_pib = next(
            fact for fact in facts
            if fact["indicator"].casefold() == "pib"
            and "2023" in fact["period"]
            and fact["measure_type"] == "volume"
        )
        self.assertEqual(nominal_pib["value"], "48 294")
        self.assertEqual(real_pib["value"], "47 092")
        self.assertEqual(nominal_pib["unit"], "milliards de FCFA")
        self.assertEqual(nominal_pib["source_method"], "spreadsheet_matrix")
        self.assertIn("Feuille", nominal_pib["qualifier"])

    def test_quarter_series_creates_traceable_total_and_variation(self) -> None:
        chunker = object.__new__(HierarchicalChunker)
        base = {
            "page": 17,
            "table_id": "page-17-period-table-1",
            "row_id": "agriculture-vivriere",
            "indicator": "AGRICULTURE VIVRIERE",
            "unit": "milliards de FCFA",
            "territory": "Cote d'Ivoire",
            "measure_type": "nominal",
            "domain": "Comptes nationaux",
            "source_method": "pdf_layout_rotated",
        }
        values = ("816,4", "823,4", "853,2", "864,6")
        facts = [
            {
                **base,
                "period": f"T{quarter}-2025",
                "value": value,
                "normalized_value": value.replace(",", "."),
            }
            for quarter, value in enumerate(values, start=1)
        ]
        chunks = chunker._derived_table_fact_chunks(
            "quarterly", {
                "url": "/quarterly.pdf", "title": "Comptes trimestriels",
                "doc_type": "pdf", "category": "anstat", "date": "2026",
            },
            "agriculture-vivriere", facts, 0,
        )
        annual = next(c for c in chunks if c.source_method == "derived_annual_sum")
        q3_q4 = next(
            c for c in chunks
            if c.source_method == "derived_quarter_change"
            and "T3 2025" in c.indicator and "T4 2025" in c.indicator
        )
        self.assertEqual(annual.value, "3357,6")
        self.assertEqual(annual.period, "2025")
        self.assertIn("816,4 + 823,4 + 853,2 + 864,6", annual.parent_text)
        self.assertEqual(q3_q4.value, "1,3")
        self.assertEqual(q3_q4.unit, "%")
        self.assertEqual(q3_q4.page, 17)
        self.assertEqual(q3_q4.territory, "Cote d'Ivoire")

    def test_multi_period_question_prefers_fact_covering_both_quarters(self) -> None:
        query = (
            "Comment le PIB a-t-il varie entre le troisieme et le "
            "quatrieme trimestre 2025 ?"
        )
        query_periods = {
            key for key in period_keys(query) if key.startswith("quarter:")
        }
        derived_periods = period_keys("T3 2025 à T4 2025")
        raw_periods = period_keys("T4 2025")
        self.assertEqual(query_periods, {
            "quarter:3:2025", "quarter:4:2025",
        })
        self.assertTrue(query_periods <= derived_periods)
        self.assertFalse(query_periods <= raw_periods)

    def test_matching_period_and_indicator_receive_the_largest_bonus(self) -> None:
        config = SimpleNamespace(
            table_fact_base_boost=0.005,
            table_fact_period_boost=0.030,
            table_fact_period_mismatch_penalty=0.080,
            table_fact_indicator_boost=0.020,
            table_fact_min_indicator_coverage=0.45,
            table_fact_focus_boost=0.100,
        )
        exact = _table_fact_bonus(
            "Combien d'hommes comptait la Côte d'Ivoire en 2025 ?",
            {
                "content_type": "table_fact",
                "indicator": "Population homme",
                "period": "2025",
            },
            config,
        )
        wrong_period = _table_fact_bonus(
            "Combien d'hommes comptait la Côte d'Ivoire en 2025 ?",
            {
                "content_type": "table_fact",
                "indicator": "Population homme",
                "period": "2024",
            },
            config,
        )
        prose = _table_fact_bonus(
            "Combien d'hommes comptait la Côte d'Ivoire en 2025 ?",
            {"content_type": "text"},
            config,
        )

        self.assertGreater(exact, wrong_period)
        self.assertLess(wrong_period, prose)

        requested_measure = _table_fact_bonus(
            "Combien d'hommes comptait la Côte d'Ivoire en 2025 ?",
            {
                "content_type": "table_fact",
                "indicator": "Population homme",
                "period": "2025",
            },
            config,
        )
        geographic_total = _table_fact_bonus(
            "Combien d'hommes comptait la Côte d'Ivoire en 2025 ?",
            {
                "content_type": "table_fact",
                "indicator": "Population de la Côte d'Ivoire",
                "period": "2025",
            },
            config,
        )
        self.assertGreater(requested_measure, geographic_total)
        wrong_year_measure = _table_fact_bonus(
            "Combien d'hommes comptait la Côte d'Ivoire en 2025 ?",
            {
                "content_type": "table_fact",
                "indicator": "Hommes",
                "period": "2015",
            },
            config,
        )
        self.assertLess(wrong_year_measure, requested_measure)

    def test_month_and_year_header_is_one_period(self) -> None:
        def word(text: str, x0: float, top: float, width: float = 38.0) -> dict:
            return {
                "text": text,
                "x0": x0,
                "x1": x0 + width,
                "top": top,
                "bottom": top + 8.0,
            }

        words = [
            word("Indicateur", 40, 20, 70),
            word("Unité", 215, 20, 40),
            word("Décembre", 300, 20, 64),
            word("2024", 370, 20, 38),
            word("Décembre", 470, 20, 64),
            word("2025", 540, 20, 38),
            word("Production", 40, 40, 72),
            word("industrielle", 116, 40, 76),
            word("%", 225, 40, 10),
            word("4,2", 345, 40, 28),
            word("6,8", 515, 40, 28),
        ]
        facts = extract_layout_table_facts(words, page_number=1)

        self.assertEqual(
            [(fact["period"], fact["value"]) for fact in facts],
            [("Décembre 2024", "4,2"), ("Décembre 2025", "6,8")],
        )

    def test_indicator_fragment_in_unit_column_is_reattached(self) -> None:
        def word(text: str, x0: float, top: float, width: float = 38.0) -> dict:
            return {
                "text": text,
                "x0": x0,
                "x1": x0 + width,
                "top": top,
                "bottom": top + 8.0,
            }

        words = [
            word("Indicateur", 40, 20, 70),
            word("Unité", 200, 20, 40),
            word("2024", 300, 20, 38),
            word("2025", 400, 20, 38),
            word("FORMATION", 40, 40, 72),
            word("BRUTE", 116, 40, 48),
            word("DE", 168, 40, 20),
            word("CAPITAL", 191, 40, 52),
            word("FIXE", 246, 40, 34),
            word("6 976", 294, 40, 48),
            word("7 140", 394, 40, 48),
        ]
        facts = extract_layout_table_facts(
            words,
            page_number=19,
            default_unit="milliards de FCFA",
        )

        self.assertEqual(len(facts), 2)
        self.assertEqual(facts[0]["indicator"], "FORMATION BRUTE DE CAPITAL FIXE")
        self.assertEqual(facts[0]["unit"], "milliards de FCFA")
        self.assertNotEqual(facts[0]["unit"], "FIXE")

    def test_indicator_text_cannot_be_accepted_as_unit(self) -> None:
        self.assertFalse(is_plausible_table_fact({
            "indicator": "EGRENAGE ET SERVICES ANNEXES A L AGRICULTURE ET",
            "period": "2025",
            "value": "411",
            "unit": "A L ELEVAGE",
        }))
        self.assertTrue(is_plausible_table_fact({
            "indicator": "EGRENAGE ET SERVICES ANNEXES A L AGRICULTURE ET A L ELEVAGE",
            "period": "2025",
            "value": "411",
            "unit": "milliards de FCFA",
        }))

    def test_numeric_table_weight_cannot_be_an_indicator(self) -> None:
        self.assertFalse(is_plausible_table_fact({
            "indicator": "10 000",
            "period": "dec-20",
            "value": "107,5",
            "unit": "%",
        }))
        self.assertTrue(is_plausible_table_fact({
            "indicator": "IPC 10 000",
            "period": "dec-20",
            "value": "107,5",
            "unit": "%",
        }))

    def test_isolated_connector_in_unit_column_rejects_the_row(self) -> None:
        def word(text: str, x0: float, top: float, width: float = 38.0) -> dict:
            return {
                "text": text,
                "x0": x0,
                "x1": x0 + width,
                "top": top,
                "bottom": top + 8.0,
            }

        words = [
            word("Indicateur", 40, 20, 70),
            word("Unité", 200, 20, 40),
            word("2024", 320, 20),
            word("2025", 440, 20),
            word("Articles", 40, 40, 55),
            word("voyage", 100, 40, 50),
            word("de", 210, 40, 18),
            word("118,7", 314, 40, 42),
            word("121,4", 434, 40, 42),
        ]
        self.assertEqual(
            extract_layout_table_facts(
                words,
                page_number=8,
                default_unit="indice",
            ),
            [],
        )

    def test_numeric_data_row_is_not_a_temporal_header(self) -> None:
        self.assertTrue(is_temporal_header_row([
            "ANNEES", "2015 CD", "2016 CD", "2017 CD", "2018 CD", "2019 CD",
        ]))
        self.assertFalse(is_temporal_header_row([
            "Taxes nettes de subvention", "2000", "2080", "2287", "2411", "2714,4",
        ]))

    def test_matrix_uses_last_label_as_indicator_and_keeps_context(self) -> None:
        rows = [
            ["", "", "1996", "1997", "1998"],
            [
                "COMPTES RETROPOLES DE 1996 A 2015 SELON LE SCN 2008 BASE 2015",
                "CROISSANCE DU PIB", "", "6,7%", "3,4%",
            ],
        ]
        facts = extract_matrix_table_facts(rows, page_number=1, default_unit="%")
        self.assertEqual([fact["period"] for fact in facts], ["1997", "1998"])
        self.assertEqual({fact["indicator"] for fact in facts}, {"CROISSANCE DU PIB"})
        self.assertTrue(all("COMPTES RETROPOLES" in fact["qualifier"] for fact in facts))
        self.assertTrue(all(is_plausible_table_fact(fact) for fact in facts))

    def test_structured_channel_retrieves_rare_indicator_before_generic_scope(self) -> None:
        def fact(
            chunk_id: str,
            parent_id: str,
            indicator: str,
            period: str,
            value: str,
        ) -> Chunk:
            return Chunk(
                chunk_id=chunk_id,
                parent_id=parent_id,
                url="/api/source-files/statistiques.pdf",
                title="Chiffres clés 2025",
                section=f"Page 10 > {indicator}",
                doc_type="pdf",
                depth=0,
                date="2025",
                category="anstat",
                text=f"Indicateur : {indicator} | Période : {period} | Valeur : {value}",
                parent_text="Série statistique",
                content_type="table_fact",
                page=10,
                row_id=parent_id,
                indicator=indicator,
                period=period,
                value=value,
                domain="Démographie",
            )

        chunks = [
            fact("total-2025", "total", "Population de la Côte d'Ivoire", "2025", "32 757 857"),
            fact("men-2024", "men", "Population homme", "2024", "16 600 000"),
            fact("men-2025", "men", "Population homme", "2025", "17 103 895"),
            fact("women-2025", "women", "Population femme", "2025", "15 653 962"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            store = BM25Store(Path(directory) / "bm25.pkl")
            store.build(chunks)
            results = store.structured_fact_candidates(
                "Combien d'hommes comptait la Côte d'Ivoire en 2025 ?",
                top_k=3,
            )

        self.assertTrue(results)
        self.assertEqual(results[0][0], "men-2025")
        self.assertNotIn("men-2024", [chunk_id for chunk_id, _ in results])

    def test_structured_channel_matches_month_alias_and_year(self) -> None:
        chunk = Chunk(
            chunk_id="dec-25",
            parent_id="industrial-production",
            url="/api/source-files/ippi.pdf",
            title="Production industrielle",
            section="Page 4",
            doc_type="pdf",
            depth=0,
            date="2025",
            category="anstat",
            text="Production industrielle | déc-25 | 16,8 %",
            parent_text="Série",
            content_type="table_fact",
            page=4,
            row_id="industrial-production",
            indicator="Taux de variation de la production industrielle",
            period="déc-25",
            value="16,8",
            unit="%",
        )
        with tempfile.TemporaryDirectory() as directory:
            store = BM25Store(Path(directory) / "bm25.pkl")
            store.build([chunk])
            results = store.structured_fact_candidates(
                "Quel est le taux de variation de la production industrielle en décembre 2025 ?",
                top_k=3,
            )

        self.assertEqual([chunk_id for chunk_id, _ in results], ["dec-25"])

    def test_cumulative_month_header_extracts_second_table_on_same_page(self) -> None:
        words = [
            _word("RUBRIQUES", 10, 10),
            _word("12", 100, 10), _word("mois-24(en", 114, 10),
            _word("12", 200, 10), _word("mois-25", 214, 10),
            _word("SOLDE", 10, 30), _word("BUDGETAIRE", 45, 30),
            _word("GLOBAL", 10, 40),
            _word("-2093,68", 105, 40), _word("-1724,70", 205, 40),
        ]
        facts = extract_layout_table_facts(
            words,
            page_number=12,
            min_periods=2,
            default_unit="milliards de FCFA",
            territory="Cote d'Ivoire",
        )
        self.assertEqual(
            [(fact["period"], fact["value"]) for fact in facts],
            [("12 mois-24", "-2093,68"), ("12 mois-25", "-1724,70")],
        )

    def test_annual_total_is_not_crowded_out_by_quarter_cells(self) -> None:
        def fact(
            chunk_id: str, period: str, value: str, *, derived: bool = False,
        ) -> Chunk:
            return Chunk(
                chunk_id=chunk_id,
                parent_id=("annual" if derived else "quarterly-series"),
                url="/api/source-files/accounts.pdf",
                title="Comptes nationaux",
                section="Agriculture vivriere",
                doc_type="pdf", depth=0, date="2026", category="anstat",
                text=f"Agriculture vivriere | {period} | {value}",
                parent_text="Serie statistique",
                content_type=("derived_fact" if derived else "table_fact"),
                page=17,
                row_id=("annual" if derived else "quarterly-series"),
                indicator=(
                    "Total annuel de AGRICULTURE VIVRIERE"
                    if derived else "AGRICULTURE VIVRIERE"
                ),
                period=period, value=value, unit="milliards de FCFA",
                measure_type="nominal",
                source_method=("derived_annual_sum" if derived else "pdf_layout"),
            )

        chunks = [
            fact("q1", "T1 2025", "816,4"),
            fact("q2", "T2 2025", "823,4"),
            fact("q3", "T3 2025", "853,2"),
            fact("q4", "T4 2025", "864,6"),
            fact("annual", "2025", "3357,6", derived=True),
        ]
        with tempfile.TemporaryDirectory() as directory:
            store = BM25Store(Path(directory) / "bm25.pkl")
            store.build(chunks)
            results = store.structured_fact_candidates(
                "Quelle etait la valeur nominale totale de l'agriculture "
                "vivriere sur l'ensemble de l'annee 2025 ?",
                top_k=2,
            )
        self.assertEqual(results[0][0], "annual")

    def test_acronym_and_measure_intent_prefer_absolute_pib(self) -> None:
        def fact(chunk_id: str, indicator: str, value: str) -> Chunk:
            return Chunk(
                chunk_id=chunk_id, parent_id=chunk_id,
                url="/api/source-files/chiffres-cles.pdf",
                title="Chiffres cles", section=indicator, doc_type="pdf",
                depth=0, date="2025", category="anstat",
                text=f"{indicator} | 2023 | {value}",
                parent_text="Serie statistique", content_type="table_fact",
                page=21, row_id=chunk_id, indicator=indicator,
                period="2023", value=value,
            )

        with tempfile.TemporaryDirectory() as directory:
            store = BM25Store(Path(directory) / "bm25.pkl")
            store.build([
                fact("ratio", "En % du PIB", "-5,2"),
                fact("absolute", "Produit Interieur Brut (PIB)", "48 294"),
            ])
            results = store.structured_fact_candidates(
                "Quel etait le PIB de la Cote d'Ivoire en 2023 ?", top_k=2,
            )

        self.assertEqual(results[0][0], "absolute")
        self.assertGreater(
            structured_measure_alignment(
                "Quel etait le PIB en 2023 ?", "Produit Interieur Brut (PIB)",
            ),
            structured_measure_alignment(
                "Quel etait le PIB en 2023 ?", "En % du PIB",
            ),
        )

    def test_academic_year_prefers_ending_year_total(self) -> None:
        def fact(chunk_id: str, period: str, value: str) -> Chunk:
            return Chunk(
                chunk_id=chunk_id, parent_id="primary-enrolment",
                url="/api/source-files/chiffres-cles.pdf",
                title="Education", section="Effectifs inscrits au primaire",
                doc_type="pdf", depth=0, date="2025", category="anstat",
                text=f"Effectifs inscrits au primaire | {period} | {value}",
                parent_text="Serie statistique", content_type="table_fact",
                page=25, row_id="primary-enrolment",
                indicator="Acces a l'education (Effectifs inscrits au primaire)",
                period=period, value=value, unit="Nombre",
            )

        with tempfile.TemporaryDirectory() as directory:
            store = BM25Store(Path(directory) / "bm25.pkl")
            store.build([
                fact("school-2023", "2023", "4 632 727"),
                fact("school-2024", "2024", "4 817 661"),
            ])
            results = store.structured_fact_candidates(
                "Combien d'eleves etaient inscrits au primaire en 2023-2024 ?",
                top_k=2,
            )

        self.assertEqual([chunk_id for chunk_id, _ in results], [
            "school-2024", "school-2023",
        ])

    def test_specific_subject_beats_generic_rate_label(self) -> None:
        question = "Quel taux de croissance etait prevu pour le secteur primaire en 2020 ?"
        self.assertGreater(
            structured_measure_alignment(question, "Secteur primaire"),
            structured_measure_alignment(question, "Taux de croissance du"),
        )

    def test_comparison_question_prefers_explicit_relation_over_cell(self) -> None:
        question = (
            "A quel taux la prevision 2020 etait-elle comparee "
            "dans la situation de reference ?"
        )
        explicit = {
            "content_type": "text",
            "title": "Cadrage macroeconomique",
            "section": "Previsions 2020",
            "text": "La croissance serait de 1,8 % contre 7,2 % en situation de reference.",
            "parent_text": "",
        }
        unrelated_cell = {
            "content_type": "table_fact",
            "title": "Chiffres cles", "section": "PIB",
            "text": "Taux de croissance du PIB | 2020 | 0,7",
            "parent_text": "",
        }
        self.assertTrue(_is_comparative_numeric_request(question))
        self.assertGreater(
            _comparison_relation_bonus(question, explicit),
            _comparison_relation_bonus(question, unrelated_cell),
        )
        mode, instruction = PromptBuilder._answer_instruction(question)
        self.assertEqual(mode, "numeric")
        self.assertIn("relation", instruction)

    def test_named_economic_measure_uses_concise_numeric_mode(self) -> None:
        mode, instruction = PromptBuilder._answer_instruction(
            "Quel était le PIB de la Côte d'Ivoire en 2023 ?",
        )
        self.assertEqual(mode, "numeric")
        self.assertIn("première phrase", instruction)

    def test_exact_badge_label_is_extracted_without_interpretation(self) -> None:
        chunks = [SimpleNamespace(
            parent_text="Kossi NEROMA, 3x Kaggle Master.\nSuite du passage.",
            text="",
        )]
        self.assertEqual(
            PromptBuilder._exact_badge_labels(chunks),
            ["Kossi NEROMA, 3x Kaggle Master"],
        )

    def test_academic_year_uses_the_selected_end_year_cell(self) -> None:
        chunks = [SimpleNamespace(
            content_type="table_fact",
            period="2024",
        )]
        instruction = PromptBuilder._academic_period_instruction(
            "Combien d'élèves étaient inscrits au primaire en 2023-2024 ?",
            chunks,
        )
        self.assertIn("une seule", instruction.casefold())
        self.assertIn("colonne de fin 2024", instruction)
        self.assertIn("Ne présente surtout pas", instruction)

    def test_calendar_interval_does_not_trigger_academic_mapping(self) -> None:
        chunks = [SimpleNamespace(
            content_type="table_fact",
            period="2024",
        )]
        instruction = PromptBuilder._academic_period_instruction(
            "Comment le PIB a-t-il évolué entre 2023-2024 ?",
            chunks,
        )
        self.assertEqual(instruction, "")

    def test_ratio_uses_two_compatible_cells_from_same_table(self) -> None:
        def fact(
            chunk_id: str, parent_id: str, indicator: str, value: str,
            *, page: int, unit: str,
        ) -> Chunk:
            return Chunk(
                chunk_id=chunk_id, parent_id=parent_id, url="/chiffres-cles.pdf",
                title="Chiffres cles", section=f"Page {page}", doc_type="pdf",
                depth=0, date="2025", category="anstat",
                text=f"{indicator} | 2023 | {value} {unit}",
                parent_text="Serie statistique", content_type="table_fact",
                page=page, row_id=parent_id, indicator=indicator,
                period="2023", value=value, unit=unit,
            )

        chunks = [
            fact("exports", "row-exports", "Exportation (Mds XOF)", "11 712", page=21, unit="En milliards"),
            fact("pib", "row-pib", "Produit Interieur Brut (PIB)", "48 294", page=21, unit="En milliards"),
            fact("tonnes", "row-tonnes", "Exportations (tonnes)", "22 998", page=30, unit="tonnes"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            store = BM25Store(Path(directory) / "bm25.pkl")
            store.build(chunks)
            components = store.structured_ratio_components(
                "Quelle etait la part des exportations dans le PIB en 2023 ?",
            )
        self.assertEqual([chunk_id for chunk_id, _ in components], ["exports", "pib"])
        metadata = {
            chunk.chunk_id: {
                "chunk_id": chunk.chunk_id, "url": chunk.url,
                "title": chunk.title, "section": chunk.section,
                "doc_type": chunk.doc_type, "category": chunk.category,
                "date": chunk.date, "page": chunk.page,
                "indicator": chunk.indicator, "period": chunk.period,
                "value": chunk.value, "unit": chunk.unit,
            }
            for chunk in chunks
        }
        derived = _derived_ratio_result(
            [chunk_id for chunk_id, _ in components], metadata,
        )
        self.assertIsNotNone(derived)
        self.assertEqual(derived.value, "24,3")
        self.assertEqual(derived.unit, "%")

    def test_explicit_cross_document_ratio_requires_compatible_facts(self) -> None:
        deficit = Chunk(
            chunk_id="deficit", parent_id="budget-row", url="/budget.pdf",
            title="Tableau de bord", section="Page 12", doc_type="pdf",
            depth=0, date="2025", category="anstat",
            text="Solde budgetaire global | 2025 | -1 400 milliards de FCFA",
            parent_text="Budget", content_type="table_fact", page=12,
            row_id="budget-row", indicator="Solde budgetaire global",
            period="12 mois-25", value="-1 400", unit="milliards de FCFA",
            territory="Cote d'Ivoire", measure_type="nominal",
        )
        pib = Chunk(
            chunk_id="annual-pib", parent_id="pib-derived", url="/pib.pdf",
            title="Comptes nationaux", section="Page 17", doc_type="pdf",
            depth=0, date="2026", category="anstat",
            text="Total annuel du PIB nominal | 2025 | 48 300 milliards de FCFA",
            parent_text="Somme de T1 a T4", content_type="derived_fact", page=17,
            row_id="pib-derived", indicator="Total annuel du PIB nominal",
            period="2025", value="48 300", unit="milliards de FCFA",
            territory="Cote d'Ivoire", measure_type="nominal",
            source_method="derived_annual_sum",
        )
        with tempfile.TemporaryDirectory() as directory:
            store = BM25Store(Path(directory) / "bm25.pkl")
            store.build([deficit, pib])
            components = store.structured_ratio_components(
                "En rapportant le deficit budgetaire 2025 au PIB nominal "
                "annuel 2025, quelle part du PIB representait-il ?",
            )
        self.assertEqual(
            [chunk_id for chunk_id, _ in components],
            ["deficit", "annual-pib"],
        )
        metadata = {
            chunk.chunk_id: {
                "chunk_id": chunk.chunk_id, "url": chunk.url,
                "title": chunk.title, "section": chunk.section,
                "doc_type": chunk.doc_type, "category": chunk.category,
                "date": chunk.date, "page": chunk.page,
                "indicator": chunk.indicator, "period": chunk.period,
                "value": chunk.value, "unit": chunk.unit,
                "territory": chunk.territory,
                "measure_type": chunk.measure_type,
            }
            for chunk in (deficit, pib)
        }
        derived = _derived_ratio_result(
            ["deficit", "annual-pib"], metadata,
            allow_cross_document=True,
            use_absolute_numerator=True,
        )
        self.assertIsNotNone(derived)
        self.assertEqual(derived.value, "2,9")
        self.assertIn("valeur absolue", derived.parent_text)

        rejected = _derived_ratio_result(
            ["deficit", "annual-pib"], metadata,
            allow_cross_document=False,
        )
        self.assertIsNone(rejected)

    def test_corrupted_shifted_table_fact_is_rejected(self) -> None:
        self.assertFalse(is_plausible_table_fact({
            "indicator": "Total Universite Sous Tutelle",
            "period": "2026", "value": "2 855 3455 641",
            "unit": "Publique 600 Du Mesrs",
        }))
        self.assertTrue(is_plausible_table_fact({
            "indicator": "Population homme", "period": "2025",
            "value": "17 103 895", "unit": "Nombre",
        }))

    def test_conflicting_values_make_the_whole_table_row_invalid(self) -> None:
        base = {
            "page": 4, "row_id": "population", "indicator": "Population homme",
            "unit": "Nombre", "territory": "Cote d'Ivoire",
            "measure_type": "effectif",
        }
        self.assertTrue(is_plausible_table_row([
            {**base, "period": "2024", "value": "16 600 000"},
            {**base, "period": "2025", "value": "17 103 895"},
        ]))
        self.assertFalse(is_plausible_table_row([
            {**base, "period": "2025", "value": "17 103 895"},
            {**base, "period": "2025", "value": "15 653 962"},
        ]))

    def test_corrupted_table_fact_does_not_consume_sparse_top_k(self) -> None:
        corrupt = Chunk(
            chunk_id="corrupt", parent_id="row-corrupt", url="/corrupt.pdf",
            title="Universite", section="Universite", doc_type="pdf", depth=0,
            date="2026", category="anstat", text="Universite 2026 2 855 3455 641",
            parent_text="Tableau OCR", content_type="table_fact", page=1,
            row_id="row-corrupt", indicator="Total Universite Sous Tutelle",
            period="2026", value="2 855 3455 641", unit="Publique 600 Du Mesrs",
        )
        valid = Chunk(
            chunk_id="valid", parent_id="row-valid", url="/valid.pdf",
            title="Universite", section="Universite", doc_type="pdf", depth=0,
            date="2026", category="anstat", text="Universite 2026 12 000",
            parent_text="Tableau valide", content_type="table_fact", page=1,
            row_id="row-valid", indicator="Effectif universitaire",
            period="2026", value="12 000", unit="Nombre",
        )
        with tempfile.TemporaryDirectory() as directory:
            store = BM25Store(Path(directory) / "bm25.pkl")
            store.build([corrupt, valid])
            results = store.search("effectif universite 2026", top_k=2)
        self.assertEqual([chunk_id for chunk_id, _ in results], ["valid"])

    def test_reference_score_requires_value_period_relation(self) -> None:
        question = "Quel était le rapport de masculinité en 2025 ?"
        reference = "109,3 hommes pour 100 femmes."
        correct = "En 2025, le rapport était de 109,3 hommes pour 100 femmes."
        wrong_year = "En 2024, le rapport était de 109,3 hommes pour 100 femmes."

        self.assertGreaterEqual(
            reference_relation_score(question, reference, correct), 0.80,
        )
        self.assertLess(
            reference_relation_score(question, reference, wrong_year), 0.50,
        )

    def test_reference_score_accepts_signed_form_of_a_decrease(self) -> None:
        score = reference_relation_score(
            "Quelle baisse des exportations etait prevue en 2020 ?",
            "10,0 %.",
            "La baisse prevue en 2020 etait de -10,0 %.",
        )
        self.assertGreaterEqual(score, 0.80)

    def test_compact_numeric_reference_must_be_in_lead_answer(self) -> None:
        question = "A quel taux la prevision etait-elle comparee en 2020 ?"
        reference = "7,2 %."
        misleading = (
            "Le scénario retenait 7,3 % en 2020. "
            "Une autre source mentionne toutefois 7,2 %."
        )
        correct = "La comparaison portait sur 7,2 % en 2020 [1]."
        self.assertGreaterEqual(
            reference_relation_score(question, reference, misleading), 0.80,
        )
        self.assertLess(
            lead_reference_relation_score(question, reference, misleading), 0.80,
        )
        self.assertGreaterEqual(
            lead_reference_relation_score(question, reference, correct), 0.80,
        )

    def test_reference_score_tolerates_very_long_ocr_numbers(self) -> None:
        score = reference_relation_score(
            "Quel etait le taux en 2023 ?", "6,5 %.",
            "En 2023, le taux etait de 6,5 %. Bruit OCR 123456789012345678901234567890123456.",
        )
        self.assertGreaterEqual(score, 0.80)

    def test_text_reference_does_not_fail_on_french_articles(self) -> None:
        score = reference_relation_score(
            "Quel diplome detient Gomez Agou ?", "Le diplome ISE 2007.",
            "Gomez Agou est titulaire d'un diplome ISE obtenu en 2007.",
        )
        self.assertGreaterEqual(score, 0.80)

    def test_abstention_detection_accepts_natural_variants(self) -> None:
        self.assertTrue(is_documentary_abstention(
            "Cette donnée n'est pas indiquée dans les sources disponibles."
        ))
        self.assertTrue(is_documentary_abstention(
            "Les documents fournis ne permettent pas de le déterminer."
        ))
        self.assertTrue(is_documentary_abstention(
            "Les sources disponibles ne mentionnent aucun chiffre spécifique pour fin 2025."
        ))
        self.assertFalse(is_documentary_abstention(
            "Le rapport de masculinité était de 109,3 en 2025."
        ))


if __name__ == "__main__":
    unittest.main()
