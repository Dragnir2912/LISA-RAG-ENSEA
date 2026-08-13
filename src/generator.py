"""Génération de réponse via Ollama (100% local), avec prompt système strict
("réponds uniquement à partir du contexte fourni") et extraction systématique
des citations [n] -> source vérifiable (passage, URL, titre, score).
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import time
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass

import ollama

from config import GenerationConfig
from conversation import (
    compact_history,
    evidence_sentences,
    is_collection_request,
    is_exhaustive_request,
    is_membership_request,
)
from retriever import RetrievedChunk

logger = logging.getLogger(__name__)


def _response_field(response, name: str, default=0):
    """Lit un champ Ollama sur les réponses dict ou objets des différentes versions."""
    if isinstance(response, dict):
        return response.get(name, default)
    return getattr(response, name, default)


def _message_field(response, name: str, default=""):
    """Lit un champ du message Ollama sur les réponses dict ou objets."""
    message = _response_field(response, "message", {})
    return _response_field(message, name, default)


def _reasoning_summary_key(reasoning: str) -> str:
    """Classe la réflexion par opération afin de produire un résumé UI localisé.

    Aucun second appel au LLM n'est nécessaire : le résumé reste instantané,
    stable et ne consomme aucun token supplémentaire.
    """
    normalized = unicodedata.normalize("NFKC", reasoning).casefold()
    categories = (
        (
            "calculation",
            r"\b(calcul|calculate|computed?|taux|rate|variation|pourcentage|percentage|formula|formule)\b",
        ),
        (
            "chronology",
            r"\b(chronolog|succession|timeline|date|dates|période|periode|year|année|annee)\b",
        ),
        (
            "enumeration",
            r"\b(exhausti|liste|list|count|nombre|membres?|members?|finalistes?|clubs?|associations?)\b",
        ),
        (
            "membership",
            r"\b(appartient|membership|affiliation|associé|associe|associated|partenaire|partner)\b",
        ),
        (
            "sources",
            r"\b(sources?|citations?|références?|references?|passages?|documents?|evidence|preuves?)\b",
        ),
    )
    for key, pattern in categories:
        if re.search(pattern, normalized):
            return key
    return "general"

SYSTEM_PROMPT = """Tu es l'assistant officiel de l'ENSEA d'Abidjan (École Nationale \
Supérieure de Statistique et d'Économie Appliquée). Tu réponds uniquement à partir \
des extraits de documents fournis dans le contexte ci-dessous.

Règles strictes :
1. N'utilise QUE les informations présentes dans le contexte. N'invente jamais rien.
2. Si le contexte ne contient pas la réponse, dis explicitement : \
"Je ne trouve pas cette information dans les sources documentaires disponibles." \
Ne tente pas de deviner.
3. Place une citation [n] juste après chaque phrase factuelle, où n est le numéro \
du passage qui contient explicitement l'information. Pour plusieurs sources, écris [1][2].
4. Ne cite jamais un passage uniquement parce que son titre semble pertinent. \
Si aucun passage ne prouve une affirmation, ne formule pas cette affirmation.
5. Réponds en français dans un style naturel, sobre et professionnel. Écris comme un \
assistant compétent qui connaît le dossier, pas comme un rapport automatique. Donne la \
réponse dès la première phrase, puis seulement les précisions utiles.
6. Choisis la mise en forme qui rend la réponse la plus lisible : un paragraphe pour \
un fait simple, une liste courte pour plusieurs éléments, et de petits intertitres \
uniquement lorsqu'une explication comporte réellement plusieurs parties.
7. N'écris jamais de titres mécaniques comme "Synthèse", "Réponse", "Liste des \
éléments attestés par le contexte" ou "Citation". Ne parle pas du "contexte fourni". \
Si la couverture documentaire est incertaine, dis simplement : "Les sources disponibles \
permettent d'identifier..." ou précise sobrement la limite en fin de réponse.
8. Ne recopie pas les extraits des sources et n'ajoute pas de ligne "Citation :" : \
les preuves sont déjà accessibles avec les références [n]. Ne donne une citation \
textuelle que si l'utilisateur demande explicitement les mots exacts.
9. Supprime tout remplissage : pas de généralités évidentes, de compliments, de conclusion \
creuse ni d'affirmation comme "a joué un rôle clé" si les sources n'en donnent pas la preuve. \
Fusionne les doublons et ne répète pas la même information dans l'introduction et la liste. \
Dans une liste, chaque entité ne doit apparaître qu'une seule fois. Après le dernier élément \
distinct documenté, termine immédiatement la réponse.
10. Pour une question au pluriel, couvre tous les éléments distincts réellement documentés. \
Pour chaque élément, combine en une ou deux phrases le nom et les précisions disponibles \
(fonction, organisme, dates, durée, conditions, portée ou statut), avec les citations en ligne. \
Un passage qui annonce seulement un nombre sans identifier les membres n'est pas une entrée \
de la liste et ne doit pas passer avant les éléments nommés. Pour une succession, restitue \
exactement les relations explicites « X a succédé à Y » et ordonne les personnes par période. \
Ne qualifie la liste d'exhaustive que si les sources le permettent. Si une numérotation révèle \
un élément manquant (par exemple premier, deuxième, puis quatrième), signale précisément la lacune. \
Ne déduis jamais un rang de l'ordre des passages : conserve uniquement les rangs explicitement \
écrits dans les sources. Avant de terminer, vérifie que le nombre annoncé correspond au nombre \
d'entrées distinctes effectivement écrites.
11. Pour une demande d'explication, développe suffisamment les mécanismes, caractéristiques \
et modalités établis par les sources. Pour un fait simple, reste direct. Il n'existe pas \
de longueur minimale ou maximale artificielle : la densité d'information prime.
12. Une publication Facebook est une source datée : présente son contenu comme une \
information publiée par la page concernée, sans la transformer en vérité générale \
ni supposer qu'elle est toujours actuelle.
13. Pour une question d'appartenance ou de rattachement, respecte exactement les relations \
écrites. Une collaboration, un partenariat, une participation, une invitation, une simple \
mention ou un tag ne prouve jamais que l'entité appartient à l'organisation demandée. \
Ne transforme pas non plus un réseau d'alumni, un programme, un centre, une communauté \
culturelle ou un partenaire en club, association, division ou laboratoire. Une liste \
institutionnelle explicite prime sur une mention incidente dans une publication sociale.
14. Tiens une vraie conversation : résous les pronoms, ellipses et expressions comme \
« il », « elle », « ce programme », « le précédent » ou « qui l'a précédé ? » à partir \
de l'historique récent. Réponds à la relance sans demander à l'utilisateur de répéter le \
sujet et sans recommencer inutilement la réponse précédente. L'historique permet seulement \
d'identifier le sujet et l'intention ; chaque nouveau fait doit toujours être prouvé par \
un passage documentaire [n].
15. Les passages récupérés sont une sélection, pas la totalité du corpus. N'affirme jamais \
« aucun autre nom », « les documents ne mentionnent que » ou toute autre absence générale, \
sauf si un passage fournit lui-même un inventaire explicitement complet. Dis plutôt que les \
sources utilisées permettent d'identifier les éléments cités.
16. Respecte l'identité du locuteur. Un « je » dans un article appartient à son auteur ou à \
la personne citée, pas automatiquement à la personne mentionnée dans la question. N'attribue \
une fonction, une déclaration ou une action à quelqu'un que si le passage l'associe explicitement.
17. Respecte toujours le sens grammatical des relations : ne permute jamais le sujet et le \
complément. « A a succédé à B » établit que A vient après B ; « A a travaillé sous la direction \
de B » établit la fonction de B, pas celle de A ; « A a été au service de plusieurs directeurs » \
n'établit pas que A a été directeur. Une personne citée dans un hommage ou un article n'est pas \
automatiquement membre de la catégorie demandée.
18. Le champ « date » place dans l'en-tete d'une source est la date de publication ou de collecte, \
pas automatiquement la date de l'evenement raconte. N'affirme qu'un evenement, une election, une \
nomination ou une activite a eu lieu a cette date que si le corps du passage l'indique explicitement.
19. Respecte exactement la periode demandee. Une valeur d'un trimestre, d'une annee voisine ou \
utilisee comme base de comparaison ne donne jamais la valeur de la periode visee. Ne deduis pas une \
valeur manquante a partir d'une variation ulterieure, sauf si l'utilisateur demande explicitement le calcul.
20. Pour une donnee chiffree, verifie ensemble l'indicateur, la population, la periode, l'unite et le \
statut (realise, estime ou prevu). Une prevision ne remplace pas une realisation, et inversement. Si deux \
passages donnent des valeurs de portee differente, distingue-les au lieu de les fusionner.
21. Conserve le sens exact des labels, grades et badges. Une formule comme « 3x Master », un rang ou \
un sigle ne prouve pas trois victoires, trois mandats ou trois diplomes : reprends le label tel qu'il est \
ecrit si les sources n'en donnent pas elles-memes l'explication.
"""


@dataclass
class Citation:
    index: int
    passage: str
    url: str
    title: str
    section: str
    similarity_score: float
    rerank_score: float


@dataclass
class RAGResponse:
    answer: str
    citations: list[Citation]


class PromptBuilder:
    """Construit un prompt borné en conservant la zone pertinente des parents."""

    def __init__(self, cfg: GenerationConfig, tokenizer=None) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.last_stats: dict[str, int | str] = {}
        self.last_passages: dict[int, str] = {}
        self._query_focused_passages = 0

    @staticmethod
    def _exact_badge_labels(chunks: list[RetrievedChunk]) -> list[str]:
        """Repère les labels compacts dont la paraphrase changerait le sens."""
        labels: list[str] = []
        pattern = re.compile(
            r"(?im)^\s*([A-ZÀ-ÖØ-Ý][^\n,]{1,80}),\s*"
            r"(\d+\s*[xX×]\s*[A-Za-zÀ-ÖØ-öø-ÿ][^\n.]{1,60})\.?\s*$"
        )
        for chunk in chunks:
            for match in pattern.finditer(chunk.parent_text or chunk.text):
                label = f"{match.group(1).strip()}, {match.group(2).strip()}"
                if label not in labels:
                    labels.append(label)
        return labels[:4]

    @staticmethod
    def _academic_period_instruction(
        query: str,
        chunks: list[RetrievedChunk],
    ) -> str:
        """Explique au générateur la convention des années scolaires.

        Les tableaux administratifs rangent fréquemment une année scolaire A-B
        dans la colonne B. Le retriever applique déjà cette préférence. Cette
        instruction empêche ensuite le LLM de retransformer la période scolaire
        en deux observations annuelles concurrentes.
        """
        normalized = unicodedata.normalize("NFKD", query.casefold())
        normalized = "".join(
            character for character in normalized
            if not unicodedata.combining(character)
        )
        academic_scope = bool(re.search(
            r"\b(?:annee scolaire|annee academique|scolaris|eleve|etudiant|"
            r"inscrit|primaire|secondaire|universit|enseignement)\w*\b",
            normalized,
        ))
        year_range = re.search(
            r"\b((?:19|20)\d{2})\s*[-–—/]\s*((?:19|20)\d{2}|\d{2})\b",
            normalized,
        )
        if not academic_scope or not year_range:
            return ""

        start_year = year_range.group(1)
        raw_end_year = year_range.group(2)
        end_year = (
            raw_end_year
            if len(raw_end_year) == 4
            else start_year[:2] + raw_end_year
        )
        selected_end_year = any(
            chunk.content_type == "table_fact"
            and str(chunk.period).strip() == end_year
            for chunk in chunks
        )
        if not selected_end_year:
            return (
                f"Période scolaire : « {start_year}-{end_year} » désigne une seule "
                "année scolaire, jamais deux observations annuelles à énumérer. "
                "N'associe une valeur que si le contexte établit sa convention de période.\n"
            )
        return (
            f"Correspondance temporelle vérifiée : la question vise l'année scolaire "
            f"{start_year}-{end_year}, qui constitue une seule période, et le fait "
            "tabulaire sélectionné la représente "
            f"par sa colonne de fin {end_year}. Réponds avec cette valeur de période "
            f"{end_year} comme valeur de l'ensemble de l'année scolaire "
            f"{start_year}-{end_year}. Ne présente surtout pas {start_year} et "
            f"{end_year} comme deux réponses annuelles distinctes.\n"
        )

    @staticmethod
    def _answer_instruction(query: str) -> tuple[str, str]:
        """Adapte la profondeur sans demander au modèle d'inventer du contenu."""
        normalized = unicodedata.normalize("NFKC", query).casefold()
        if re.search(
            r"\b(compar(?:é|ée|e|ee|er|aison)|contre|écart|différence|"
            r"scénario de référence|situation de référence|par rapport)\b",
            normalized,
        ) and re.search(r"\b(taux|part|ratio|valeur|niveau|nombre)\b", normalized):
            return "numeric", (
                "La question demande un terme précis d'une comparaison. Repère le passage "
                "qui exprime explicitement cette relation (par exemple « X contre Y en "
                "situation de référence ») et donne uniquement la valeur attachée au membre "
                "demandé, avec sa citation [n]. Une prévision générale issue d'un autre "
                "document et une réalisation tabulaire de la même année ne répondent pas à "
                "cette relation. N'énumère pas les valeurs concurrentes."
            )
        if re.search(
            r"\b(qui (?:était|est|sera)|quel(?:le)? (?:expert|personne|nom|diplôme|diplome|"
            r"fonction|poste|titre))\b",
            normalized,
        ):
            return "direct", (
                "Identifie directement la personne, le diplôme, la fonction ou le titre demandé "
                "avec sa référence [n], puis ajoute seulement les précisions explicitement liées "
                "à cette identité. Ne déduis aucun autre diplôme, rôle ou accomplissement à partir "
                "d'un titre de civilité, d'un sigle ou d'un badge."
            )
        if re.search(
            r"\b(combien|de combien|quel(?:le)? (?:était|est|sera) (?:le |la )?"
            r"(?:taux|niveau|nombre|effectif|part|proportion|ratio|rapport|valeur|"
            r"pib|population|densité|densite|production|consommation|investissement|"
            r"exportation|importation|solde|dette|revenu|dépense|depense|recette|"
            r"indice|prix|montant|durée|duree|temps|distance))\b",
            normalized,
        ):
            return "numeric", (
                "Donne la valeur demandée dès la première phrase avec sa période, son unité "
                "et sa référence [n]. Ajoute au plus une précision utile si elle est prouvée. "
                "N'énumère pas les années, taux ou indicateurs voisins et ne calcule rien sauf "
                "si la question demande explicitement un calcul."
            )
        if re.search(
            r"\b(quels|quelles|lesquels|lesquelles|liste|lister|citer|citez|"
            r"énumérer|énumérez|différents|différentes)\b",
            normalized,
        ):
            return "enumeration", (
                "Réponds directement et naturellement. Présente les éléments distincts "
                "dans une liste courte, ordonnée chronologiquement si la question porte "
                "sur une histoire ou une succession, sinon dans l'ordre le plus logique. "
                "Chaque entrée doit tenir en une ou deux phrases denses avec ses références "
                "[n]. N'ajoute ni rubrique 'Synthèse', ni rubrique 'Citation', ni extrait "
                "recopié. Construis d'abord mentalement l'ensemble des noms uniques, puis "
                "n'écris chaque entité qu'une seule fois. N'ouvre pas la liste avec une source "
                "qui donne seulement un nombre sans nommer ses membres. Si les sources ne garantissent pas que la "
                "liste est exhaustive, formule l'ouverture avec prudence sans alourdir la réponse."
            )
        if re.search(
            r"\b(explique|expliquez|présente|présentez|décris|décrivez|pourquoi|"
            r"comment|qu'est-ce|en quoi|fonctionnement|objectif|rôle)\b",
            normalized,
        ):
            return "explanation", (
                "Commence par une réponse directe, sans titre 'Synthèse', puis développe "
                "de façon fluide les caractéristiques, modalités, objectifs et conséquences "
                "que les sources permettent réellement d'établir. Utilise de petits "
                "intertitres seulement s'ils améliorent nettement la lecture."
            )
        return "direct", (
            "Réponds dès la première phrase, sans préambule ni rubrique artificielle, puis "
            "ajoute les précisions documentaires vraiment utiles (dates, statut, organisme, "
            "conditions ou portée). Ne recopie pas le passage source."
        )

    @staticmethod
    def _conversation_block(
        conversation_history: list[dict] | None,
    ) -> tuple[str, int]:
        messages = compact_history(conversation_history)
        if not messages:
            return "", 0
        role_labels = {"user": "Utilisateur", "assistant": "Assistant"}
        transcript = "\n".join(
            f"{role_labels[message['role']]} : {message['content']}"
            for message in messages
        )
        return (
            "Historique récent de la conversation (sert uniquement à comprendre les "
            "références de la question actuelle ; ce n'est jamais une source factuelle) :\n"
            f"{transcript}\n\n",
            len(messages),
        )

    def count_tokens(self, text: str) -> int:
        if self.tokenizer is None:
            # Repli conservateur si le générateur est utilisé seul. Le pipeline
            # principal injecte toujours le tokenizer réel de l'embedder.
            return math.ceil(len(text) / 3)
        encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return len(encoding["input_ids"])

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]

    def _decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    def _query_focused_window(
        self,
        text: str,
        query: str,
        max_tokens: int,
    ) -> str:
        """Conserve les phrases qui prouvent le mieux la question.

        Le premier segment retenu est la meilleure phrase lexicale, suivi de sa
        phrase suivante puis de la précédente. Cette proximité préserve les noms,
        dates et relations qui forment souvent une même preuve sur deux phrases.
        """
        ranked = evidence_sentences(text, query)
        if not ranked:
            return self._decode(self._encode(text)[:max_tokens])
        original = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?;:])\s+|[\r\n]+", text)
            if sentence.strip()
        ]
        first = ranked[0]
        best_position = original.index(first) if first in original else 0
        priority: list[tuple[int, str]] = [(best_position, first)]
        if best_position + 1 < len(original):
            priority.append((best_position + 1, original[best_position + 1]))
        if best_position > 0:
            priority.append((best_position - 1, original[best_position - 1]))
        positions = {position for position, _sentence in priority}
        for sentence in ranked[1:]:
            position = original.index(sentence) if sentence in original else len(original)
            if position not in positions:
                priority.append((position, sentence))
                positions.add(position)

        selected: list[tuple[int, str]] = []
        remaining = max_tokens
        for priority_index, (position, sentence) in enumerate(priority):
            token_ids = self._encode(sentence)
            if len(token_ids) <= remaining:
                selected.append((position, sentence))
                remaining -= len(token_ids)
            elif remaining >= 12:
                # Pour une collection, une phrase très longue peut contenir un
                # premier groupe de noms et la suivante compléter la série.
                # On lui réserve donc un petit quart du budget au lieu de laisser
                # la première phrase absorber tout l'extrait.
                reserve = 0
                if (
                    priority_index == 0
                    and is_collection_request(query)
                    and len(priority) > 1
                ):
                    next_ids = self._encode(priority[1][1])
                    reserve = min(len(next_ids), max(16, max_tokens // 4))
                take = max(12, remaining - reserve)
                selected.append((position, self._decode(token_ids[:take])))
                remaining -= take
            if remaining < 12:
                break
        selected.sort(key=lambda item: item[0])
        self._query_focused_passages += 1
        return " ".join(sentence for _position, sentence in selected).strip()

    def _relevant_window(
        self,
        parent: str,
        child: str,
        max_tokens: int,
        query: str,
    ) -> str:
        """Tronque un parent autour du chunk enfant qui a déclenché le retrieval."""
        if self.count_tokens(parent) <= max_tokens:
            return parent

        if self.tokenizer is None:
            max_chars = max_tokens * 3
            position = parent.find(child)
            if position < 0:
                return parent[:max_chars]
            start = max(0, position - max(0, max_chars - len(child)) // 2)
            return parent[start:start + max_chars]

        position = parent.find(child)
        if position < 0:
            return self._query_focused_window(parent, query, max_tokens)

        before_ids = self._encode(parent[:position])
        child_ids = self._encode(child)
        after_ids = self._encode(parent[position + len(child):])

        if len(child_ids) >= max_tokens:
            return self._query_focused_window(child, query, max_tokens)

        remaining = max_tokens - len(child_ids)
        before_budget = min(len(before_ids), remaining // 2)
        after_budget = min(len(after_ids), remaining - before_budget)
        unused = remaining - before_budget - after_budget
        if unused and len(before_ids) > before_budget:
            before_budget += min(unused, len(before_ids) - before_budget)
        token_ids = (
            before_ids[-before_budget:] if before_budget else []
        ) + child_ids + after_ids[:after_budget]
        return self._decode(token_ids)

    def _build_context(
        self,
        chunks: list[RetrievedChunk],
        budget: int,
        *,
        query: str,
        preserve_all: bool = False,
    ) -> tuple[str, dict[int, str]]:
        blocks: list[str] = []
        passages: dict[int, str] = {}
        fair_passage_cap = self.cfg.max_passage_tokens
        if preserve_all and chunks:
            header_tokens = sum(
                self.count_tokens(
                    f"[{index}] (source : {chunk.title} ; url : {chunk.url} ; "
                    f"type : {chunk.doc_type})\n"
                )
                for index, chunk in enumerate(chunks, start=1)
            )
            separator_tokens = max(0, len(chunks) - 1) * self.count_tokens("\n\n")
            # Marge par source pour les dates, séparateurs et petits écarts entre
            # le tokenizer local et celui d'Ollama. Elle évite que le dernier
            # résultat soit exclu pour quelques tokens.
            safety_tokens = 16 * len(chunks)
            passage_budget = max(
                0,
                budget - header_tokens - separator_tokens - safety_tokens,
            )
            fair_passage_cap = max(32, passage_budget // len(chunks))
        for index, chunk in enumerate(chunks, start=1):
            source_details = [f"source : {chunk.title}", f"url : {chunk.url}"]
            if chunk.date:
                source_details.append(f"date : {chunk.date}")
            source_details.append(f"type : {chunk.doc_type}")
            if chunk.content_type == "table_fact":
                source_details.append("preuve : fait tabulaire structure")
                if chunk.page:
                    source_details.append(f"page : {chunk.page}")
            if chunk.coverage_reserved:
                source_details.append("statut : élément de couverture")
            header = f"[{index}] ({' ; '.join(source_details)})\n"
            current = "\n\n".join(blocks)
            separator = "\n\n" if blocks else ""
            fixed_tokens = self.count_tokens(current + separator + header)
            available = budget - fixed_tokens
            if available <= 0:
                break

            passage_budget = min(
                self.cfg.max_passage_tokens,
                fair_passage_cap,
                available,
            )
            passage = self._relevant_window(
                chunk.parent_text, chunk.text, passage_budget, query,
            )
            candidate = current + separator + header + passage

            # Les tokenizers sous-mots ne sont pas parfaitement additifs autour
            # des séparateurs. On réduit légèrement la fenêtre si nécessaire.
            overflow = self.count_tokens(candidate) - budget
            if overflow > 0:
                passage_budget = max(1, passage_budget - overflow - 4)
                passage = self._relevant_window(
                    chunk.parent_text, chunk.text, passage_budget, query,
                )
                candidate = current + separator + header + passage
            if self.count_tokens(candidate) > budget:
                break
            blocks.append(header + passage)
            passages[index] = passage
        return "\n\n".join(blocks), passages

    def build_messages(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        conversation_history: list[dict] | None = None,
    ) -> list[dict]:
        answer_mode, answer_instruction = self._answer_instruction(query)
        membership_instruction = ""
        if is_membership_request(query):
            membership_instruction = (
                "Vérification d'appartenance : n'inclus une entité que si un passage "
                "affirme explicitement son rattachement à l'organisation visée. "
                "Les mots 'avec', 'en collaboration avec', 'partenaire', 'invité', "
                "'participant' et les simples mentions ou tags indiquent une relation "
                "extérieure, pas une appartenance. Respecte le type exact de chaque "
                "entité et privilégie les inventaires institutionnels explicites.\n"
            )
        coverage_indices = [
            index for index, chunk in enumerate(chunks, start=1)
            if chunk.coverage_reserved
        ]
        coverage_instruction = ""
        if coverage_indices:
            labels = ", ".join(f"[{index}]" for index in coverage_indices)
            coverage_instruction = (
                "Contrôle de couverture : les passages " + labels + " ont été "
                "retenus parce qu'ils contiennent la catégorie et la relation demandées, "
                "puis ont été confirmés par le reranker. Examine chacun d'eux. Toute entité distincte dont "
                "le passage affirme explicitement l'appartenance à cette collection doit "
                "apparaître exactement une fois dans la réponse. Une entité seulement "
                "mentionnée, invitée ou partenaire ne doit pas être incluse. "
                "Si plusieurs passages concernent la même entité, fusionne-les dans une "
                "seule entrée et rassemble leurs citations.\n"
            )
        structured_fact_instruction = ""
        if any(
            chunk.content_type in {"table_fact", "derived_fact"}
            for chunk in chunks
        ):
            structured_fact_instruction = (
                "Donnees chiffrees : privilegie les faits tabulaires structures dont "
                "l'indicateur et la periode correspondent exactement a la question. "
                "Reprends leur valeur et leur unite sans calcul ni changement d'annee. "
                "Un passage marque comme calcul deterministe contient deja le ratio obtenu "
                "a partir de deux cellules compatibles : reprends ce resultat sans le recalculer. "
                "Si un tel fait repond explicitement a la demande, ne dis pas que l'information "
                "est absente. Ignore les lignes portant sur une mesure, une population ou une "
                "modalite differente.\n"
            )
        academic_period_instruction = self._academic_period_instruction(
            query, chunks,
        )
        exact_labels = self._exact_badge_labels(chunks)
        exact_label_instruction = ""
        if exact_labels:
            quoted_labels = " ; ".join(f"« {label} »" for label in exact_labels)
            exact_label_instruction = (
                "Labels exacts détectés dans les preuves : " + quoted_labels + ". "
                "Si tu mentionnes l'entité concernée, reproduis son label mot pour mot. "
                "Ne transforme jamais « Nx/× titre » en nombre de victoires, de diplômes, "
                "de mandats ou de statuts obtenus.\n"
            )
        conversation_block, conversation_turns = self._conversation_block(
            conversation_history,
        )
        prompt_without_context = (
            f"{conversation_block}Contexte documentaire :\n\n\n"
            f"Question actuelle : {query}\n\n"
            f"{answer_instruction}\n"
            f"{membership_instruction}"
            f"{coverage_instruction}"
            f"{structured_fact_instruction}"
            f"{academic_period_instruction}"
            f"{exact_label_instruction}"
            "Réponds en citant les passages utilisés avec [n]."
        )
        input_limit = max(0, self.cfg.num_ctx - self.cfg.reserved_output_tokens)
        fixed_tokens = self.count_tokens(SYSTEM_PROMPT) + self.count_tokens(
            prompt_without_context
        )
        context_budget = min(
            self.cfg.context_budget_tokens,
            max(0, input_limit - fixed_tokens),
        )
        self._query_focused_passages = 0
        context, self.last_passages = self._build_context(
            chunks,
            context_budget,
            query=query,
            preserve_all=(
                is_exhaustive_request(query) or is_collection_request(query)
            ),
        )
        user_prompt = (
            f"{conversation_block}Contexte documentaire :\n{context}\n\n"
            f"Question actuelle : {query}\n\n"
            f"{answer_instruction}\n"
            f"{membership_instruction}"
            f"{coverage_instruction}"
            f"{structured_fact_instruction}"
            f"{academic_period_instruction}"
            f"{exact_label_instruction}"
            "Réponds en citant les passages utilisés avec [n]."
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prompt_tokens = sum(self.count_tokens(message["content"]) for message in messages)
        self.last_stats = {
            "retrieved_chunks": len(chunks),
            "included_chunks": len(self.last_passages),
            "context_tokens": self.count_tokens(context),
            "prompt_tokens": prompt_tokens,
            "input_token_limit": input_limit,
            "reserved_output_tokens": self.cfg.reserved_output_tokens,
            "answer_mode": answer_mode,
            "membership_validation": bool(membership_instruction),
            "coverage_required_chunks": len(coverage_indices),
            "query_focused_passages": self._query_focused_passages,
            "conversation_turns": conversation_turns,
        }
        return messages


def _citation_indices(answer: str) -> set[int]:
    """Reconnaît [1], [1][2], [1, 2] et [1; 2] sans capturer d'autres nombres."""
    indices: set[int] = set()
    for group in re.findall(r"\[([0-9\s,;]+)\]", answer):
        indices.update(int(value) for value in re.findall(r"\d+", group))
    return indices


_LIST_ITEM_PREFIX = re.compile(r"^\s*(?:[-*+•]\s+|\d+[.)]\s+)")


def _list_item_signature(line: str) -> str | None:
    """Signature stable d'une puce pour bloquer les boucles verbatim du LLM."""
    if not _LIST_ITEM_PREFIX.match(line):
        return None
    content = _LIST_ITEM_PREFIX.sub("", line, count=1)
    content = re.sub(r"\[[0-9\s,;]+\]", "", content)
    normalized = unicodedata.normalize("NFKD", content.casefold())
    normalized = "".join(
        character for character in normalized
        if not unicodedata.combining(character)
    )
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized).strip()
    return normalized or None


def _deduplicate_list_lines(answer: str) -> tuple[str, int]:
    """Supprime les puces strictement répétées sans toucher aux faits distincts."""
    seen: set[str] = set()
    kept: list[str] = []
    removed = 0
    for line in answer.splitlines(keepends=True):
        signature = _list_item_signature(line)
        if signature and signature in seen:
            removed += 1
            continue
        if signature:
            seen.add(signature)
        kept.append(line)
    return "".join(kept).strip(), removed


def extract_citations(
    answer: str,
    chunks: list[RetrievedChunk],
    prompt_passages: dict[int, str] | None = None,
) -> list[Citation]:
    """Ne conserve que les citations [n] effectivement mentionnées dans la réponse,
    et restitue exactement le passage documentaire montré au générateur."""
    cited_indices = _citation_indices(answer)
    valid_indices = set(range(1, len(chunks) + 1))
    invalid_indices = cited_indices - valid_indices
    if invalid_indices:
        logger.warning("Citations inexistantes ignorées : %s", sorted(invalid_indices))

    citations = []
    for i, chunk in enumerate(chunks, start=1):
        if i in cited_indices:
            citations.append(Citation(
                index=i,
                passage=(prompt_passages or {}).get(i, chunk.text),
                url=chunk.url,
                title=chunk.title,
                section=chunk.section, similarity_score=chunk.dense_score,
                rerank_score=chunk.rerank_score,
            ))
    return citations


class OllamaGenerator:
    def __init__(self, cfg: GenerationConfig, tokenizer=None) -> None:
        self.cfg = cfg
        self.prompt_builder = PromptBuilder(cfg, tokenizer=tokenizer)
        self.last_prompt_stats: dict[str, int | str] = {}
        self.last_generation_stats: dict[str, float | int | bool] = {}
        self.last_reasoning: dict[str, str | float | bool] = {}
        self._response_cache: OrderedDict[
            tuple[str, tuple[str, ...]],
            tuple[
                RAGResponse,
                dict[str, int | str],
                dict[str, str | float | bool],
            ],
        ] = OrderedDict()

    @staticmethod
    def _normalize_query(query: str) -> str:
        normalized = unicodedata.normalize("NFKC", query).casefold()
        return " ".join(normalized.split())

    def _cache_key(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        conversation_history: list[dict] | None = None,
        thinking: bool | None = None,
    ) -> tuple[str, tuple[str, ...]]:
        # Le cache dépend aussi du comportement de génération. Une modification
        # du prompt ou des paramètres ne doit jamais resservir une ancienne réponse.
        _answer_mode, answer_instruction = self.prompt_builder._answer_instruction(query)
        thinking_enabled = self.cfg.thinking if thinking is None else bool(thinking)
        generation_behavior = "\0".join((
            SYSTEM_PROMPT,
            answer_instruction,
            self.cfg.model_name,
            str(self.cfg.temperature),
            str(thinking_enabled),
            str(self.cfg.max_output_tokens),
            str(self.cfg.thinking_output_tokens),
            str(self.cfg.direct_output_tokens),
            str(self.cfg.numeric_output_tokens),
            str(self.cfg.explanation_output_tokens),
            str(self.cfg.enumeration_output_tokens),
            str(self.cfg.repeat_penalty),
            str(self.cfg.repeat_last_n),
            str(self.cfg.max_consecutive_duplicate_list_items),
        ))
        behavior_signature = hashlib.sha256(
            generation_behavior.encode("utf-8")
        ).hexdigest()[:16]
        history_content = "\0".join(
            f"{message['role']}:{message['content']}"
            for message in compact_history(conversation_history)
        )
        history_signature = hashlib.sha256(
            history_content.encode("utf-8")
        ).hexdigest()[:12]
        chunk_signatures = []
        for chunk in chunks:
            content = "\0".join(
                (
                    chunk.chunk_id, chunk.url, chunk.title, chunk.text,
                    chunk.parent_text, str(chunk.coverage_reserved),
                )
            )
            chunk_signatures.append(
                hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
            )
        normalized_query = self._normalize_query(query)
        return (
            f"{behavior_signature}:{history_signature}:{normalized_query}",
            tuple(chunk_signatures),
        )

    def _cache_get(
        self,
        key: tuple[str, tuple[str, ...]],
        *,
        thinking_enabled: bool,
    ) -> RAGResponse | None:
        cached = self._response_cache.pop(key, None)
        if cached is None:
            return None
        response, prompt_stats, reasoning = cached
        self._response_cache[key] = cached
        self.last_prompt_stats = dict(prompt_stats)
        self.last_reasoning = dict(reasoning)
        self.last_generation_stats = {
            "cache_hit": True,
            "total_seconds": 0.0,
            "ollama_prompt_tokens": 0,
            "generated_tokens": 0,
            "tokens_per_second": 0.0,
            "load_seconds": 0.0,
            "prompt_eval_seconds": 0.0,
            "generation_eval_seconds": 0.0,
            "server_total_seconds": 0.0,
            "unaccounted_seconds": 0.0,
            "thinking_enabled": thinking_enabled,
        }
        logger.info("Réponse servie depuis le cache exact.")
        return response

    def _cache_put(
        self,
        key: tuple[str, tuple[str, ...]],
        response: RAGResponse,
    ) -> None:
        if self.cfg.response_cache_size <= 0:
            return
        self._response_cache[key] = (
            response,
            dict(self.last_prompt_stats),
            dict(self.last_reasoning),
        )
        self._response_cache.move_to_end(key)
        while len(self._response_cache) > self.cfg.response_cache_size:
            self._response_cache.popitem(last=False)

    def _runtime_num_ctx(self) -> int:
        """Dimensionne le cache KV par paliers de 1024 selon le prompt réel."""
        required = (
            self.last_prompt_stats.get("prompt_tokens", 0)
            + self.cfg.reserved_output_tokens
        )
        rounded = math.ceil(required / 1024) * 1024
        return min(self.cfg.num_ctx, max(self.cfg.min_num_ctx, rounded))

    def prepare_messages(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        conversation_history: list[dict] | None = None,
    ) -> list[dict]:
        messages = self.prompt_builder.build_messages(
            query, chunks, conversation_history=conversation_history,
        )
        self.last_prompt_stats = dict(self.prompt_builder.last_stats)
        self.last_prompt_stats["ollama_num_ctx"] = self._runtime_num_ctx()
        logger.info("Budget du prompt : %s", self.last_prompt_stats)
        return messages

    @staticmethod
    def _with_thinking_guard(messages: list[dict]) -> list[dict]:
        """Empêche le modèle de dépenser tout son budget à relire les consignes."""
        guard = {
            "role": "system",
            "content": (
                "Raisonne de façon concise sans recopier ni commenter les consignes. "
                "Passe impérativement à la réponse finale avant d'épuiser le budget."
            ),
        }
        if messages and messages[0].get("role") == "system":
            return [messages[0], guard, *messages[1:]]
        return [guard, *messages]

    def _runtime_num_predict(self, *, thinking_enabled: bool = False) -> int:
        if thinking_enabled:
            return min(
                int(self.cfg.thinking_output_tokens),
                int(self.cfg.reserved_output_tokens),
            )
        mode = self.last_prompt_stats.get("answer_mode", "direct")
        requested = {
            "numeric": self.cfg.numeric_output_tokens,
            "direct": self.cfg.direct_output_tokens,
            "explanation": self.cfg.explanation_output_tokens,
            "enumeration": self.cfg.enumeration_output_tokens,
        }.get(str(mode), self.cfg.max_output_tokens)
        return min(int(requested), int(self.cfg.max_output_tokens))

    def _ollama_options(self, *, num_predict: int) -> dict[str, int | float]:
        """Construit les options communes, avec forçage CPU facultatif."""
        options: dict[str, int | float] = {
            "temperature": self.cfg.temperature,
            "num_ctx": self.last_prompt_stats["ollama_num_ctx"],
            "num_predict": num_predict,
            "num_batch": self.cfg.num_batch,
            "repeat_penalty": self.cfg.repeat_penalty,
            "repeat_last_n": self.cfg.repeat_last_n,
        }
        if self.cfg.num_gpu is not None:
            options["num_gpu"] = int(self.cfg.num_gpu)
        return options

    def _record_empty_generation(self, *, thinking_enabled: bool) -> None:
        """Évite de réutiliser les métriques de la requête précédente en cas d'abstention."""
        self.last_prompt_stats = {
            "retrieved_chunks": 0,
            "included_chunks": 0,
            "prompt_tokens": 0,
        }
        self.last_generation_stats = {
            "cache_hit": False,
            "total_seconds": 0.0,
            "ollama_prompt_tokens": 0,
            "generated_tokens": 0,
            "tokens_per_second": 0.0,
            "thinking_enabled": thinking_enabled,
        }
        self.last_reasoning = {}

    def _record_reasoning(
        self,
        content: str,
        *,
        seconds: float = 0.0,
        truncated: bool = False,
    ) -> None:
        cleaned = content.strip()
        self.last_reasoning = (
            {
                "content": cleaned,
                "summary_key": _reasoning_summary_key(cleaned),
                "seconds": round(max(0.0, seconds), 4),
                "truncated": bool(truncated),
            }
            if cleaned
            else {}
        )

    def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        conversation_history: list[dict] | None = None,
        *,
        thinking: bool | None = None,
    ) -> RAGResponse:
        thinking_enabled = self.cfg.thinking if thinking is None else bool(thinking)
        if not chunks:
            self._record_empty_generation(thinking_enabled=thinking_enabled)
            return RAGResponse(
                answer="Je ne trouve pas cette information dans les sources documentaires disponibles.",
                citations=[],
            )

        cache_key = self._cache_key(
            query, chunks, conversation_history, thinking=thinking_enabled,
        )
        cached_response = self._cache_get(
            cache_key, thinking_enabled=thinking_enabled,
        )
        if cached_response is not None:
            return cached_response

        self.last_reasoning = {}
        messages = self.prepare_messages(query, chunks, conversation_history)
        num_predict = self._runtime_num_predict(thinking_enabled=thinking_enabled)
        self.last_prompt_stats["max_generated_tokens"] = num_predict
        started = time.perf_counter()
        response = ollama.chat(
            model=self.cfg.model_name,
            messages=(
                self._with_thinking_guard(messages)
                if thinking_enabled else messages
            ),
            think=thinking_enabled,
            options=self._ollama_options(num_predict=num_predict),
            keep_alive=self.cfg.keep_alive,
            stream=False,
        )
        answer, repeated_lines_removed = _deduplicate_list_lines(
            str(_message_field(response, "content", "") or "")
        )
        reasoning_content = str(_message_field(response, "thinking", "") or "")
        responses = [response]
        fallback_answer_used = False
        if thinking_enabled and not answer.strip():
            fallback_answer_used = True
            logger.warning(
                "Le budget a été consommé par la réflexion ; génération finale sans réflexion."
            )
            response = ollama.chat(
                model=self.cfg.model_name,
                messages=messages,
                think=False,
                options=self._ollama_options(
                    num_predict=self._runtime_num_predict(thinking_enabled=False),
                ),
                keep_alive=self.cfg.keep_alive,
                stream=False,
            )
            responses.append(response)
            answer, repeated_lines_removed = _deduplicate_list_lines(
                str(_message_field(response, "content", "") or "")
            )
        elapsed = time.perf_counter() - started
        self._record_reasoning(
            reasoning_content,
            seconds=elapsed if reasoning_content else 0.0,
            truncated=fallback_answer_used,
        )
        prompt_eval_count = sum(
            int(_response_field(item, "prompt_eval_count", 0) or 0)
            for item in responses
        )
        eval_count = sum(
            int(_response_field(item, "eval_count", 0) or 0)
            for item in responses
        )
        eval_duration_seconds = sum(
            float(_response_field(item, "eval_duration", 0) or 0)
            for item in responses
        ) / 1_000_000_000
        prompt_eval_seconds = sum(
            float(_response_field(item, "prompt_eval_duration", 0) or 0)
            for item in responses
        ) / 1_000_000_000
        server_total_seconds = sum(
            float(_response_field(item, "total_duration", 0) or 0)
            for item in responses
        ) / 1_000_000_000
        load_seconds = sum(
            float(_response_field(item, "load_duration", 0) or 0)
            for item in responses
        ) / 1_000_000_000
        self.last_generation_stats = {
            "cache_hit": False,
            "total_seconds": elapsed,
            "ollama_prompt_tokens": prompt_eval_count,
            "generated_tokens": eval_count,
            "tokens_per_second": (
                eval_count / eval_duration_seconds if eval_duration_seconds > 0 else 0.0
            ),
            "load_seconds": load_seconds,
            "prompt_eval_seconds": prompt_eval_seconds,
            "generation_eval_seconds": eval_duration_seconds,
            "server_total_seconds": server_total_seconds,
            "unaccounted_seconds": max(
                0.0,
                elapsed - load_seconds - prompt_eval_seconds - eval_duration_seconds,
            ),
            "repeated_lines_removed": repeated_lines_removed,
            "repetition_guard_triggered": False,
            "thinking_enabled": thinking_enabled,
            "fallback_answer_used": fallback_answer_used,
            "reasoning_seconds": elapsed if reasoning_content else 0.0,
            "reasoning_characters": len(reasoning_content),
            "reasoning_truncated": fallback_answer_used,
        }
        logger.info("Génération Ollama : %s", self.last_generation_stats)
        citations = extract_citations(
            answer,
            chunks,
            prompt_passages=self.prompt_builder.last_passages,
        )
        rag_response = RAGResponse(answer=answer, citations=citations)
        self._cache_put(cache_key, rag_response)
        return rag_response

    def generate_stream(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        conversation_history: list[dict] | None = None,
        *,
        thinking: bool | None = None,
        event_mode: bool = False,
    ):
        """Génère en streaming.

        Par défaut, cette méthode conserve le contrat historique et ne renvoie que
        les fragments de réponse. ``event_mode=True`` ajoute des événements
        ``reasoning`` séparés, utilisés exclusivement par l'application web.
        """
        thinking_enabled = self.cfg.thinking if thinking is None else bool(thinking)
        if not chunks:
            self._record_empty_generation(thinking_enabled=thinking_enabled)
            abstention = "Je ne trouve pas cette information dans les sources documentaires disponibles."
            if event_mode:
                yield {"type": "token", "content": abstention}
            else:
                yield abstention
            return

        cache_key = self._cache_key(
            query, chunks, conversation_history, thinking=thinking_enabled,
        )
        cached_response = self._cache_get(
            cache_key, thinking_enabled=thinking_enabled,
        )
        if cached_response is not None:
            if event_mode and self.last_reasoning.get("content"):
                yield {
                    "type": "reasoning",
                    "content": str(self.last_reasoning["content"]),
                    "cached": True,
                }
            if event_mode:
                yield {"type": "token", "content": cached_response.answer}
            else:
                yield cached_response.answer
            return

        self.last_reasoning = {}
        messages = self.prepare_messages(query, chunks, conversation_history)
        num_predict = self._runtime_num_predict(thinking_enabled=thinking_enabled)
        self.last_prompt_stats["max_generated_tokens"] = num_predict
        started = time.perf_counter()
        stream = ollama.chat(
            model=self.cfg.model_name,
            messages=(
                self._with_thinking_guard(messages)
                if thinking_enabled else messages
            ),
            think=thinking_enabled,
            options=self._ollama_options(num_predict=num_predict),
            keep_alive=self.cfg.keep_alive,
            stream=True,
        )
        last_part = None
        answer_parts: list[str] = []
        reasoning_parts: list[str] = []
        reasoning_started_at: float | None = None
        first_answer_at: float | None = None
        pending_line = ""
        seen_list_items: set[str] = set()
        consecutive_duplicates = 0
        repeated_lines_removed = 0
        repetition_guard_triggered = False
        for part in stream:
            last_part = part
            reasoning_content = str(_message_field(part, "thinking", "") or "")
            if reasoning_content:
                if reasoning_started_at is None:
                    reasoning_started_at = time.perf_counter()
                reasoning_parts.append(reasoning_content)
                if event_mode:
                    yield {"type": "reasoning", "content": reasoning_content}
            content = str(_message_field(part, "content", "") or "")
            if content and first_answer_at is None:
                first_answer_at = time.perf_counter()
            pending_line += content
            stop_stream = False
            while "\n" in pending_line:
                line, pending_line = pending_line.split("\n", 1)
                rendered_line = line + "\n"
                signature = _list_item_signature(line)
                if signature and signature in seen_list_items:
                    repeated_lines_removed += 1
                    consecutive_duplicates += 1
                    if (
                        consecutive_duplicates
                        >= self.cfg.max_consecutive_duplicate_list_items
                    ):
                        repetition_guard_triggered = True
                        stop_stream = True
                        break
                    continue
                if signature:
                    seen_list_items.add(signature)
                if line.strip():
                    consecutive_duplicates = 0
                answer_parts.append(rendered_line)
                if event_mode:
                    yield {"type": "token", "content": rendered_line}
                else:
                    yield rendered_line
            if stop_stream:
                break
        primary_finished_at = time.perf_counter()
        primary_last_part = last_part
        metric_parts = [primary_last_part] if primary_last_part is not None else []
        fallback_answer_used = False
        if (
            thinking_enabled
            and not repetition_guard_triggered
            and not answer_parts
            and not pending_line.strip()
        ):
            fallback_answer_used = True
            pending_line = ""
            logger.warning(
                "Le streaming n'a produit que de la réflexion ; relance finale sans réflexion."
            )
            fallback_stream = ollama.chat(
                model=self.cfg.model_name,
                messages=messages,
                think=False,
                options=self._ollama_options(
                    num_predict=self._runtime_num_predict(thinking_enabled=False),
                ),
                keep_alive=self.cfg.keep_alive,
                stream=True,
            )
            last_part = None
            for part in fallback_stream:
                last_part = part
                content = str(_message_field(part, "content", "") or "")
                if content and first_answer_at is None:
                    first_answer_at = time.perf_counter()
                pending_line += content
                stop_stream = False
                while "\n" in pending_line:
                    line, pending_line = pending_line.split("\n", 1)
                    rendered_line = line + "\n"
                    signature = _list_item_signature(line)
                    if signature and signature in seen_list_items:
                        repeated_lines_removed += 1
                        consecutive_duplicates += 1
                        if (
                            consecutive_duplicates
                            >= self.cfg.max_consecutive_duplicate_list_items
                        ):
                            repetition_guard_triggered = True
                            stop_stream = True
                            break
                        continue
                    if signature:
                        seen_list_items.add(signature)
                    if line.strip():
                        consecutive_duplicates = 0
                    answer_parts.append(rendered_line)
                    if event_mode:
                        yield {"type": "token", "content": rendered_line}
                    else:
                        yield rendered_line
                if stop_stream:
                    break
            if last_part is not None:
                metric_parts.append(last_part)
        if pending_line and not repetition_guard_triggered:
            signature = _list_item_signature(pending_line)
            if signature and signature in seen_list_items:
                repeated_lines_removed += 1
            else:
                answer_parts.append(pending_line)
                if event_mode:
                    yield {"type": "token", "content": pending_line}
                else:
                    yield pending_line
        elapsed = time.perf_counter() - started
        reasoning_seconds = 0.0
        if reasoning_started_at is not None:
            reasoning_seconds = (
                (
                    primary_finished_at
                    if fallback_answer_used
                    else (first_answer_at or primary_finished_at)
                )
                - reasoning_started_at
            )
        self._record_reasoning(
            "".join(reasoning_parts),
            seconds=reasoning_seconds,
            truncated=bool(
                fallback_answer_used
                and str(_response_field(primary_last_part, "done_reason", ""))
                == "length"
            ),
        )
        eval_count = sum(
            int(_response_field(item, "eval_count", 0) or 0)
            for item in metric_parts
        )
        eval_duration_seconds = sum(
            float(_response_field(item, "eval_duration", 0) or 0)
            for item in metric_parts
        ) / 1_000_000_000
        prompt_eval_seconds = sum(
            float(_response_field(item, "prompt_eval_duration", 0) or 0)
            for item in metric_parts
        ) / 1_000_000_000
        server_total_seconds = sum(
            float(_response_field(item, "total_duration", 0) or 0)
            for item in metric_parts
        ) / 1_000_000_000
        load_seconds = sum(
            float(_response_field(item, "load_duration", 0) or 0)
            for item in metric_parts
        ) / 1_000_000_000
        self.last_generation_stats = {
            "cache_hit": False,
            "total_seconds": elapsed,
            "ollama_prompt_tokens": int(
                sum(
                    int(_response_field(item, "prompt_eval_count", 0) or 0)
                    for item in metric_parts
                )
            ),
            "generated_tokens": eval_count,
            "tokens_per_second": (
                eval_count / eval_duration_seconds if eval_duration_seconds > 0 else 0.0
            ),
            "load_seconds": load_seconds,
            "prompt_eval_seconds": prompt_eval_seconds,
            "generation_eval_seconds": eval_duration_seconds,
            "server_total_seconds": server_total_seconds,
            "unaccounted_seconds": max(
                0.0,
                elapsed - load_seconds - prompt_eval_seconds - eval_duration_seconds,
            ),
            "repeated_lines_removed": repeated_lines_removed,
            "repetition_guard_triggered": repetition_guard_triggered,
            "thinking_enabled": thinking_enabled,
            "reasoning_seconds": reasoning_seconds,
            "reasoning_characters": len("".join(reasoning_parts)),
            "fallback_answer_used": fallback_answer_used,
            "reasoning_truncated": bool(
                fallback_answer_used
                and str(_response_field(primary_last_part, "done_reason", ""))
                == "length"
            ),
        }
        logger.info("Génération Ollama streaming : %s", self.last_generation_stats)
        answer = "".join(answer_parts)
        citations = extract_citations(
            answer,
            chunks,
            prompt_passages=self.prompt_builder.last_passages,
        )
        self._cache_put(cache_key, RAGResponse(answer=answer, citations=citations))
