from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from conversation import (  # noqa: E402
    compact_history,
    contextualize_question,
    direct_conversation_reply,
    evidence_sentences,
    exhaustive_category_tokens,
    filter_unproven_ocr_roles,
    generation_history,
    is_collection_request,
    is_exhaustive_request,
    is_membership_request,
    retrieval_subject_text,
    requested_role_aliases,
)
from source_router import resolve_facebook_page_scope, route_question  # noqa: E402


class ConversationContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.history = [
            {
                "role": "user",
                "content": "Quels sont les finalistes du concours d'éloquence du CDR de cette année ?",
            },
            {
                "role": "assistant",
                "content": "Les sources disponibles identifient plusieurs finalistes.",
            },
        ]

    def test_followup_reuses_previous_user_question_and_keeps_facebook_route(self) -> None:
        query, contextualized = contextualize_question(
            "La liste est-elle exhaustive ? Je veux la liste exhaustive.",
            self.history,
        )
        self.assertTrue(contextualized)
        self.assertIn("concours d'éloquence du CDR", query)
        self.assertEqual(route_question(query).route, "facebook")

    def test_pronoun_object_followup_carries_resolved_entity_into_retrieval(self) -> None:
        history = [
            {"role": "user", "content": "Qui est le DG de l'ENSEA ?"},
            {
                "role": "assistant",
                "content": (
                    "Le Directeur Général de l'ENSEA est Dr Hugues KOUADIO "
                    "depuis 2015 [1]. Il coordonne la direction [2]."
                ),
            },
        ]
        query, contextualized = contextualize_question("Qui l'a précédé ?", history)
        self.assertTrue(contextualized)
        self.assertIn("Qui est le DG de l'ENSEA", query)
        self.assertIn("Dr Hugues KOUADIO", query)
        self.assertIn("Qui l'a précédé", query)
        self.assertNotIn("[1]", query)
        self.assertEqual(route_question(query).route, "institutional")

    def test_complete_succession_question_is_not_mistaken_for_followup(self) -> None:
        question = "Quels sont les directeurs qui se sont succédé à la tête de l'ENSEA ?"
        query, contextualized = contextualize_question(question, self.history)
        self.assertFalse(contextualized)
        self.assertEqual(query, question)

    def test_elliptical_succession_question_remains_a_followup(self) -> None:
        query, contextualized = contextualize_question(
            "Les directeurs qui se sont succédé à la tête ?", self.history,
        )
        self.assertTrue(contextualized)
        self.assertIn("Question précédente", query)

    def test_social_turns_do_not_require_document_retrieval(self) -> None:
        self.assertIsNotNone(direct_conversation_reply("Bonjour"))
        self.assertIsNotNone(direct_conversation_reply("Merci beaucoup"))
        self.assertIsNotNone(direct_conversation_reply("D'accord"))
        self.assertIsNotNone(direct_conversation_reply("Que peux-tu faire ?"))
        self.assertIsNone(
            direct_conversation_reply("Merci, et qui l'a précédé ?")
        )

    def test_social_turn_does_not_replace_the_conversation_subject(self) -> None:
        history = [
            {"role": "user", "content": "Qui est le DG de l'ENSEA ?"},
            {"role": "assistant", "content": "Il s'agit de Dr Hugues KOUADIO [1]."},
            {"role": "user", "content": "Merci"},
            {
                "role": "assistant",
                "content": "Avec plaisir. Nous pouvons poursuivre sur ce sujet.",
            },
        ]
        query, contextualized = contextualize_question("Et son prédécesseur ?", history)
        self.assertTrue(contextualized)
        self.assertIn("Qui est le DG de l'ENSEA", query)
        self.assertIn("Dr Hugues KOUADIO", query)
        self.assertNotIn("Avec plaisir", query)

    def test_independent_question_is_not_polluted_by_history(self) -> None:
        question = "Qui est le Directeur Général de l'ENSEA ?"
        query, contextualized = contextualize_question(question, self.history)
        self.assertFalse(contextualized)
        self.assertEqual(query, question)

    def test_anstat_question_uses_economic_index(self) -> None:
        self.assertEqual(
            route_question("Quel est le PIB de la Côte d'Ivoire selon l'ANSTAT ?").route,
            "economic",
        )
        self.assertEqual(
            route_question("Que disent les résultats du RGPH 2021 ?").route,
            "economic",
        )

    def test_natural_industrial_statistics_questions_use_economic_index(self) -> None:
        questions = [
            "Quel est le taux de variation de la production industrielle en Côte d'Ivoire en décembre 2025 ?",
            "Comment a évolué l'activité industrielle ivoirienne en mars 2026 ?",
            "Quelle est la production manufacturière en Côte d'Ivoire ?",
            "Les prix de production industrielle ont-ils augmenté ?",
        ]
        for question in questions:
            with self.subTest(question=question):
                self.assertEqual(route_question(question).route, "economic")

    def test_generic_ensea_rates_and_production_remain_institutional(self) -> None:
        questions = [
            "Quel est le taux d'admission à l'ENSEA ?",
            "Quelle est la production scientifique de l'ENSEA ?",
        ]
        for question in questions:
            with self.subTest(question=question):
                self.assertEqual(route_question(question).route, "institutional")

    def test_ensea_question_remains_institutional(self) -> None:
        self.assertEqual(
            route_question("Quelles sont les conditions d'admission à l'ENSEA ?").route,
            "institutional",
        )

    def test_time_bound_student_events_use_facebook(self) -> None:
        questions = [
            "Qui a remporté le concours ENSEA Éloquence 2026 ?",
            "Qui sont les finalistes du Golden Speaker cette année ?",
            "Quelle nouvelle équipe a été élue à la tête d'un club de l'ENSEA ?",
            "Quel bureau associatif a gagné les élections 2026 ?",
            (
                "Quels sont les candidats à la direction du Club de Débats "
                "et de Réflexions pour le prochain mandat ?"
            ),
        ]
        for question in questions:
            with self.subTest(question=question):
                self.assertEqual(route_question(question).route, "facebook")

    def test_configured_page_aliases_drive_routing_without_code_change(self) -> None:
        decision = route_question(
            "Quelles sont les dernières activités de Robotique ENSEA ?",
            page_aliases={"ROBOTIQUE_page": ["robotique ensea"]},
        )
        self.assertEqual(decision.route, "facebook")
        self.assertIn("ROBOTIQUE_page", decision.reason)

    def test_ambiguous_recent_ensea_events_search_both_indexes(self) -> None:
        decision = route_question(
            "Qui a remporté le dernier concours scientifique organisé à l'ENSEA en 2026 ?"
        )
        self.assertEqual(decision.route, "both")

    def test_unknown_topic_defaults_to_all_corpora(self) -> None:
        decision = route_question(
            "Quel est le palmarès du challenge NOVA organisé sur le campus ?"
        )
        self.assertEqual(decision.route, "all")
        self.assertIn("route ambiguë", decision.reason)

    def test_general_national_statistics_use_economic_index(self) -> None:
        questions = [
            "Quel était le taux de croissance démographique en 2025 ?",
            "Quel était le rapport de masculinité en 2025 ?",
            "Quel était le taux de mortalité maternelle ?",
            "Quel était le taux d'accès à l'électricité ?",
            "Quelle était la proportion de PME ?",
            "Quel était le taux net de scolarisation au primaire ?",
            "Quel était le taux de chômage des femmes ?",
            "Quel était le taux de pauvreté en milieu rural ?",
            "Quel était le montant des recettes fiscales ?",
            "Quel était le niveau de la dette publique ivoirienne ?",
        ]
        for question in questions:
            with self.subTest(question=question):
                self.assertEqual(route_question(question).route, "economic")

    def test_person_profile_diploma_does_not_exclude_social_sources(self) -> None:
        decision = route_question("Quel diplôme de l'ENSEA détient Gomez Agou ?")
        self.assertEqual(decision.route, "all")

    def test_admission_competitions_remain_institutional(self) -> None:
        questions = [
            "Quels sont les résultats du concours d'entrée à l'ENSEA en 2026 ?",
            "Quelles sont les conditions d'admission au concours de l'ENSEA ?",
        ]
        for question in questions:
            with self.subTest(question=question):
                self.assertEqual(route_question(question).route, "institutional")

    def test_followup_chain_returns_to_last_independent_question(self) -> None:
        history = self.history + [
            {"role": "user", "content": "La liste est-elle exhaustive ?"},
            {"role": "assistant", "content": "Je ne trouve pas cette information."},
        ]
        query, contextualized = contextualize_question(
            "Je veux vraiment la liste complète.", history,
        )
        self.assertTrue(contextualized)
        self.assertIn("concours d'éloquence du CDR", query)
        self.assertEqual(route_question(query).route, "facebook")

    def test_exhaustive_request_is_detected(self) -> None:
        self.assertTrue(is_exhaustive_request("Je veux la liste exhaustive."))
        self.assertTrue(is_exhaustive_request("Donne-moi tous les finalistes."))
        self.assertFalse(is_exhaustive_request("Qui a gagné la finale ?"))

    def test_plural_enumeration_is_a_collection_request(self) -> None:
        self.assertTrue(is_collection_request("Quels sont les laboratoires ?"))
        self.assertTrue(is_collection_request("Quelles accréditations sont valides ?"))
        self.assertFalse(is_collection_request("Qui dirige l'ENSEA ?"))

    def test_clubs_and_associations_use_authoritative_source(self) -> None:
        decision = route_question(
            "Quels sont les clubs et associations présents à l'ENSEA ?"
        )
        self.assertEqual(decision.route, "institutional")

    def test_membership_questions_are_detected_without_matching_events(self) -> None:
        self.assertTrue(
            is_membership_request(
                "Quels sont les clubs et associations de l'ENSEA ?"
            )
        )
        self.assertTrue(
            is_membership_request("Quels laboratoires sont associés à l'ENSEA ?")
        )
        self.assertFalse(
            is_membership_request("Quel club organise un concours d'éloquence ?")
        )

    def test_exhaustive_category_is_general_and_uses_initial_question(self) -> None:
        examples = {
            "Quels sont les finalistes du concours ?": ["finaliste"],
            "Quels sont donc les directeurs de l'ENSEA ?": ["directeur"],
            "Quelles sont exactement les accréditations de la filière ?": ["accreditation"],
            "Quelles sont les accréditations de la filière ?": ["accreditation"],
            "Quels directeurs l'ENSEA a-t-elle connus ?": ["directeur"],
            "Donne la liste exhaustive des laboratoires.": ["laboratoire"],
            "Je veux tous les partenaires.": ["partenaire"],
        }
        for question, expected in examples.items():
            with self.subTest(question=question):
                self.assertEqual(exhaustive_category_tokens(question), expected)

        contextualized = (
            "Question précédente : Quels sont les finalistes du concours ?\n"
            "Question de suivi : La liste est-elle exhaustive ?"
        )
        self.assertEqual(exhaustive_category_tokens(contextualized), ["finaliste"])

    def test_retrieval_subject_ignores_contextualization_labels_and_old_answer(self) -> None:
        contextualized = (
            "Question précédente : Quels sont les finalistes du concours ?\n"
            "Référence résolue au tour précédent : Aka et cinq autres candidats.\n"
            "Question de suivi : La liste est-elle exhaustive ?"
        )
        subject = retrieval_subject_text(contextualized)
        self.assertEqual(
            subject,
            "Quels sont les finalistes du concours ?\n"
            "La liste est-elle exhaustive ?",
        )
        self.assertNotIn("précédente", subject.casefold())
        self.assertNotIn("Aka", subject)

    def test_independent_collection_drops_old_answers_from_generation(self) -> None:
        history = [
            {"role": "user", "content": "Qui dirige l'ENSEA ?"},
            {"role": "assistant", "content": "Hugues est le seul directeur [1]."},
        ]
        selected = generation_history(
            "Quels directeurs se sont succédé à la tête de l'ENSEA ?",
            history,
            contextualized=False,
        )
        self.assertEqual(selected, [])

    def test_independent_topic_change_drops_old_factual_history(self) -> None:
        history = [
            {"role": "user", "content": "Qui dirige l'EJS ?"},
            {"role": "assistant", "content": "L'equipe NOVA cloture son mandat."},
        ]
        question = "As-tu une idee de la nouvelle equipe a la tete du CDR ?"
        _query, contextualized = contextualize_question(question, history)
        self.assertFalse(contextualized)
        self.assertEqual(
            generation_history(
                question, history, contextualized=contextualized,
            ),
            [],
        )

    def test_true_followup_keeps_conversation_history(self) -> None:
        history = [
            {"role": "user", "content": "Qui dirige l'EJS ?"},
            {"role": "assistant", "content": "Le directeur est indique [1]."},
        ]
        question = "Et qui est son adjoint ?"
        _query, contextualized = contextualize_question(question, history)
        self.assertTrue(contextualized)
        self.assertEqual(
            generation_history(
                question, history, contextualized=contextualized,
            ),
            history,
        )

    def test_followup_keeps_only_the_current_topic_segment(self) -> None:
        history = [
            {"role": "user", "content": "Qui dirige l'EJS ?"},
            {"role": "assistant", "content": "L'equipe NOVA est mentionnee."},
            {"role": "user", "content": "Quelle est la nouvelle equipe du CDR ?"},
            {"role": "assistant", "content": "Une liste candidate est annoncee."},
        ]
        question = "Et qui est son president ?"
        _query, contextualized = contextualize_question(question, history)
        selected = generation_history(
            question, history, contextualized=contextualized,
        )
        self.assertTrue(contextualized)
        self.assertEqual(selected, history[2:])
        self.assertNotIn("NOVA", " ".join(item["content"] for item in selected))

    def test_demonstrative_noun_phrase_is_a_general_followup(self) -> None:
        history = [
            {"role": "user", "content": "Quelle est la nouvelle equipe du CDR ?"},
            {"role": "assistant", "content": "Une liste candidate est annoncee."},
        ]
        question = "Quel etait le nom de cette equipe ?"
        query, contextualized = contextualize_question(question, history)
        self.assertTrue(contextualized)
        self.assertIn("CDR", query)
        self.assertEqual(route_question(query).route, "facebook")

    def test_current_year_does_not_create_a_false_followup(self) -> None:
        question = "Quels sont les finalistes du CDR cette annee ?"
        _query, contextualized = contextualize_question(question, self.history)
        self.assertFalse(contextualized)

    def test_noisy_ocr_cannot_create_a_director(self) -> None:
        context = (
            "L'equipe NOVA cloture son mandat.\n\n"
            "## Texte detecte dans les images\n"
            "- EJS BIGEA Jela Services"
        )
        filtered = filter_unproven_ocr_roles(
            context, "Qui etait le DG de l'EJS ?",
        )
        self.assertNotIn("Jela", filtered)
        self.assertEqual(
            requested_role_aliases("Qui etait le DG de l'EJS ?")[0],
            "directeur general",
        )

    def test_ocr_role_is_kept_when_explicit_in_the_same_fragment(self) -> None:
        context = (
            "Publication du bureau.\n\n"
            "## Texte detecte dans les images\n"
            "- Directrice generale : Jeanne Exemple"
        )
        filtered = filter_unproven_ocr_roles(
            context, "Qui est la directrice generale ?",
        )
        self.assertIn("Jeanne Exemple", filtered)

    def test_non_role_question_keeps_ocr_labels(self) -> None:
        context = "## Texte detecte dans les images\n- EXCELLENCIA"
        self.assertEqual(
            filter_unproven_ocr_roles(context, "Quel est le nom de l'equipe ?"),
            context,
        )

    def test_facebook_scope_is_resolved_from_configured_aliases(self) -> None:
        aliases = {
            "CDR_page": ["cdr", "club de debats et de reflexions"],
            "EJS_page": ["ejs", "ensea junior services"],
        }
        self.assertEqual(
            resolve_facebook_page_scope(
                "Quelle est la nouvelle equipe du CDR ?", aliases,
            ),
            "CDR_page",
        )
        self.assertEqual(
            resolve_facebook_page_scope(
                "Qui dirigeait l'EJS en 2025-2026 ?", aliases,
            ),
            "EJS_page",
        )

    def test_query_focused_sentence_prioritizes_relational_evidence(self) -> None:
        text = (
            "L'école célèbre son anniversaire. "
            "Koffi N'Guessan a été directeur pendant vingt ans après avoir succédé "
            "au Dr François Yattien-Amiguet. "
            "Depuis 2015, Hugues Kouadio dirige l'établissement."
        )
        ranked = evidence_sentences(
            text, "Quels directeurs se sont succédé à la tête de l'ENSEA ?",
        )
        self.assertIn("Koffi N'Guessan", ranked[0])
        self.assertIn("François Yattien-Amiguet", ranked[0])

    def test_history_is_bounded(self) -> None:
        long_history = self.history * 4
        compacted = compact_history(long_history, max_messages=3)
        self.assertEqual(len(compacted), 3)

    def test_old_citation_numbers_are_removed_from_assistant_history(self) -> None:
        compacted = compact_history([
            {"role": "assistant", "content": "Aka André [4] et Ouattara Marc [2]."},
        ])
        self.assertEqual(compacted[0]["content"], "Aka André et Ouattara Marc.")


if __name__ == "__main__":
    unittest.main()
