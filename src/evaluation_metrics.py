"""Metriques deterministes pour les evaluations du RAG ENSEA.

Les reponses de reference de la Banque 100 sont souvent tres courtes. Une
simple intersection de mots penalise injustement les variantes francaises
(``minute``/``minutes``), la ponctuation des nombres (``11,9``/``11.9``) et les
formulations d'abstention naturelles. Ce module separe donc explicitement les
nombres, les unites et le contenu lexical.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


_STOPWORDS = {
    "a", "au", "aux", "avec", "ce", "ces", "cette", "dans", "de", "des",
    "du", "elle", "en", "est", "et", "la", "le", "les", "pour", "que",
    "qui", "un", "une", "avait", "etaient", "etait", "selon", "indique",
    "base", "documentaire", "fournie", "fourni",
}
_OPTIONAL_REFERENCE_WORDS = {
    "approximation", "estimation", "estime", "estimee", "environ",
}
_UNIT_ALIASES = {
    "%": "%",
    "pourcent": "%",
    "pourcentage": "%",
    "h": "heure",
    "heure": "heure",
    "heures": "heure",
    "min": "minute",
    "minute": "minute",
    "minutes": "minute",
    "hab": "habitant",
    "habitant": "habitant",
    "habitants": "habitant",
    "homme": "homme",
    "hommes": "homme",
    "femme": "femme",
    "femmes": "femme",
    "fcfa": "fcfa",
    "xof": "fcfa",
    "milliard": "milliard",
    "milliards": "milliard",
    "million": "million",
    "millions": "million",
    "millier": "millier",
    "milliers": "millier",
    "tonne": "tonne",
    "tonnes": "tonne",
    "jour": "jour",
    "jours": "jour",
    "mois": "mois",
    "an": "annee",
    "ans": "annee",
    "annee": "annee",
    "annees": "annee",
}
_CITATION_RE = re.compile(r"\[(?:\d+)(?:\]\[\d+)*\]")
_NUMBER_RE = re.compile(
    r"(?<![\w])[-+−]?(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)"
    r"(?:[.,]\d+)?(?!\d)"
)


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text).casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _light_stem(token: str) -> str:
    token = _UNIT_ALIASES.get(token, token)
    if len(token) > 5 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    return token


def normalized_tokens(text: str) -> list[str]:
    normalized = normalize_text(_CITATION_RE.sub(" ", str(text)))
    tokens = []
    # Les heures sont souvent ecrites sans espace (``9h``). La separation des
    # lettres et chiffres les rend equivalentes a ``9 h``.
    for token in re.findall(r"%|[a-z]+|\d+", normalized):
        token = _light_stem(token)
        if token not in _STOPWORDS and (len(token) > 1 or token.isdigit() or token == "%"):
            tokens.append(token)
    return tokens


def normalized_numbers(text: str) -> list[str]:
    without_citations = _CITATION_RE.sub(" ", str(text))
    numbers = []
    for match in _NUMBER_RE.finditer(without_citations):
        value = match.group(0).replace("−", "-")
        value = value.replace("\u00a0", "").replace("\u202f", "").replace(" ", "")
        value = value.replace(",", ".")
        try:
            numeric = float(value)
        except ValueError:
            continue
        numbers.append(f"{numeric:.12g}")
    return numbers


def reference_units(text: str) -> set[str]:
    units = set()
    for token in normalized_tokens(text):
        canonical = _UNIT_ALIASES.get(token)
        if canonical:
            units.add(canonical)
    return units


@dataclass(frozen=True)
class FactCoverage:
    score: float
    numeric_recall: float | None
    lexical_recall: float | None
    unit_recall: float | None
    exact_numeric_match: bool


def reference_fact_metrics(reference: str, answer: str) -> FactCoverage:
    """Compare les faits sans confondre style et exactitude numerique."""
    expected_numbers = list(dict.fromkeys(normalized_numbers(reference)))
    observed_numbers = set(normalized_numbers(answer))
    numeric_recall = None
    if expected_numbers:
        # ``1 556 000 FCFA (soit 1 556 milliers)`` donne deux representations
        # equivalentes. Une reponse n'a pas a reproduire les deux. En revanche,
        # les nombres d'une relation sans marqueur d'equivalence (109,3 pour
        # 100) restent tous obligatoires.
        alternatives = re.split(
            r"\b(?:soit|equivalent(?:e)?(?: a)?|c est a dire)\b",
            normalize_text(reference).replace("'", " ").replace("’", " "),
        )
        alternative_numbers = [
            list(dict.fromkeys(normalized_numbers(part)))
            for part in alternatives if normalized_numbers(part)
        ]
        if len(alternative_numbers) > 1:
            numeric_recall = max(
                sum(number in observed_numbers for number in numbers) / len(numbers)
                for numbers in alternative_numbers
            )
        else:
            numeric_recall = (
                sum(number in observed_numbers for number in expected_numbers)
                / len(expected_numbers)
            )

    expected_units = reference_units(reference)
    observed_units = reference_units(answer)
    unit_recall = (
        len(expected_units & observed_units) / len(expected_units)
        if expected_units else None
    )

    expected_lexical = {
        token for token in normalized_tokens(reference)
        if not token.isdigit()
        and token not in _OPTIONAL_REFERENCE_WORDS
        and token not in expected_units
        and token not in {"%"}
    }
    observed_lexical = set(normalized_tokens(answer))
    lexical_recall = (
        len(expected_lexical & observed_lexical) / len(expected_lexical)
        if expected_lexical else None
    )

    weighted = []
    if numeric_recall is not None:
        weighted.append((numeric_recall, 0.70))
    if unit_recall is not None:
        weighted.append((unit_recall, 0.15))
    if lexical_recall is not None:
        weighted.append((lexical_recall, 0.15 if numeric_recall is not None else 1.0))
    total_weight = sum(weight for _value, weight in weighted)
    score = sum(value * weight for value, weight in weighted) / total_weight if total_weight else 0.0
    return FactCoverage(
        score=float(score),
        numeric_recall=numeric_recall,
        lexical_recall=lexical_recall,
        unit_recall=unit_recall,
        exact_numeric_match=(numeric_recall == 1.0 if numeric_recall is not None else False),
    )


def reference_fact_coverage(reference: str, answer: str) -> float:
    return reference_fact_metrics(reference, answer).score


def is_abstention(answer: str) -> bool:
    """Reconnait une absence explicite, y compris suivie de donnees historiques."""
    normalized = re.sub(r"\s+", " ", normalize_text(answer)).strip()
    normalized = normalized.replace("'", " ").replace("’", " ")
    patterns = (
        r"\bje ne trouv(?:e|ons)(?: donc)? pas\b",
        r"\binformation(?: precise| exacte| specifique)? (?:n est|est non) disponible\b",
        r"\b(?:donnee|information|valeur|chiffre|indicateur)s? (?:n est|ne sont|non) (?:pas )?(?:disponible|fourni|fournie|mentionne|mentionnee)s?\b",
        r"\baucun(?:e)? (?:donnee|information|valeur|chiffre|indicateur)[^.;]{0,200}\b(?:fourni|fournie|mentionne|mentionnee|disponible)s?\b",
        r"\b(?:sources?|documents?|extraits?|passages?)[^.;]{0,100}\bne (?:permettent|mentionnent|fournissent|contiennent|precisent) pas\b",
        r"\b(?:sources?|documents?)[^.;]{0,100}\bn informent pas (?:specifiquement|precisement)?\b",
        r"\b(?:elles|ils|ceux ci|celles ci) n informent pas (?:specifiquement|precisement)?\b",
        r"\bn incluent aucune (?:donnee|information|valeur|chiffre)[^.;]{0,100}\b",
        r"\baucun (?:indicateur|document|passage)[^.;]{0,100}\bne (?:fournit|donne|precise|mentionne)\b",
        r"\bimpossible (?:de|a) (?:determiner|identifier|calculer|etablir)\b",
        r"\bsans (?:fournir|indiquer|mentionner|preciser)[^.;]{0,80}\b(?:valeur|chiffre|donnee|information)\b",
        r"\bne (?:permet|permettent) (?:donc )?pas (?:de )?(?:repondre|determiner|identifier|calculer)\b",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def route_compatible(actual: str, expected: str) -> bool:
    """`all` couvre la bonne collection mais reste moins specifique."""
    actual = str(actual or "").strip().casefold()
    expected = str(expected or "").strip().casefold()
    if actual == expected:
        return True
    return actual == "all" and expected in {"institutional", "economic", "facebook"}


def answerable_automated_pass(
    reference: str,
    answer: str,
    *,
    citation_source_hit: bool,
    minimum_textual_coverage: float = 0.75,
) -> bool:
    metrics = reference_fact_metrics(reference, answer)
    if not citation_source_hit:
        return False
    if metrics.numeric_recall is not None:
        # Pour une reponse chiffree, toutes les valeurs de la reference doivent
        # etre presentes. Cela conserve Q6 en reponse partielle : 109,3 seul ne
        # suffit pas a exprimer « 109,3 hommes pour 100 femmes ».
        if metrics.numeric_recall < 1.0:
            return False
        if metrics.unit_recall is not None and metrics.unit_recall < 1.0:
            return False
        return True
    return metrics.score >= minimum_textual_coverage
