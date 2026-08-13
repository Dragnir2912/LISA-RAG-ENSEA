"""Extraction et normalisation de faits statistiques issus de tableaux.

Les PDF statistiques contiennent souvent des tableaux visuellement corrects mais
dont le texte extrait est aplati. Ce module exploite les coordonnees des mots
fournies par pdfplumber pour reconstruire la relation fondamentale :

    indicateur + periode + valeur + unite

La representation obtenue est volontairement generique et serialisable en JSON.
Elle peut etre utilisee par le chunker, BM25, LanceDB et le generateur sans
dependre du document ou de la question qui a motive son ajout.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Iterable, Sequence


TABLE_EXTRACTION_VERSION = 7
SPREADSHEET_EXTRACTION_VERSION = 7

_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
_QUARTER_RE = re.compile(r"^(?:t|q)[1-4](?:[-_/ ]?(?:19|20)?\d{2,4})?$", re.I)
_MONTH_RE = re.compile(
    r"^(?:janv(?:ier)?|fevr(?:ier)?|mars|avr(?:il)?|mai|juin|juil(?:let)?|"
    r"aout|sept(?:embre)?|oct(?:obre)?|nov(?:embre)?|dec(?:embre)?|"
    r"january|february|march|april|may|june|july|august|september|"
    r"october|november|december)(?:[-_/ ]?(?:19|20)?\d{2,4})?$",
    re.I,
)
_NUMERIC_RE = re.compile(
    r"[-+−]?\d[\d\s\u00a0\u202f]*(?:[.,]\d+)?(?:\s*%)?"
)
_FOOTNOTE_RE = re.compile(r"\s*(?:\([^)]{1,8}\)|\*{1,4})\s*$")
_STRICT_NUMERIC_VALUE_RE = re.compile(
    r"^[-+−]?\s*(?:\d{1,3}(?:[\s\u00a0\u202f]\d{3})+|\d+)"
    r"(?:[.,]\d+)?\s*%?$"
)

_UNIT_WORDS = {
    "%", "‰", "nombre", "effectif", "valeur", "minute", "minutes",
    "annee", "annees", "ans", "jour", "jours", "mois", "tonne", "tonnes",
    "fcfa", "xof", "milliard", "milliards", "million", "millions", "millier",
    "milliers", "indice", "ratio", "rang", "km", "km2", "km²", "hab", "hbts",
    "habitant", "habitants", "pour", "100", "000", "en", "de", "par",
}

_UNIT_TOKEN_RE = re.compile(
    r"^(?:1\s*)?(?:unite|service|kg|g|gramme|grammes|litre|litres|"
    r"tonne|tonnes|m3|m2|km|km2|km²|ha|hectare|hectares|minute|minutes|"
    r"jour|jours|mois|an|ans|annee|annees)$"
)
_AMBIGUOUS_LABEL_FRAGMENTS = {
    "a", "au", "aux", "d", "de", "des", "du", "en", "et", "ou", "par",
}


def _is_plausible_unit(text: str) -> bool:
    """Reconnaît une unité, sans accepter un fragment de libellé de ligne.

    Dans de nombreux PDF, le libellé d'un indicateur déborde visuellement dans
    la colonne « Unité ». Une règle fondée uniquement sur la longueur acceptait
    alors ``A L ELEVAGE``, ``FIXE`` ou ``D'EXPORTATION`` comme unités. Les
    unités sont ici reconnues par leur forme ; les autres fragments sont rendus
    au libellé de l'indicateur par l'extracteur.
    """
    compact = _compact(text).strip()
    if not compact:
        return False
    normalized = _ascii(compact).strip("()[] .:;")
    normalized = re.sub(r"\s+", " ", normalized)
    if compact in {"%", "‰"} or normalized in {"pourcent", "pourcentage"}:
        return True
    if normalized in {
        "nombre", "effectif", "indice", "ratio", "rang", "point", "points",
    }:
        return True
    if _UNIT_TOKEN_RE.fullmatch(normalized):
        return True
    if re.fullmatch(r"(?:1\s*)?(?:kg|g|l|ml|m|m2|m3|km|km2|ha)", normalized):
        return True
    if re.search(r"\b(?:fcfa|xof|f\s*cfa)\b", normalized):
        return True
    if re.fullmatch(
        r"(?:en\s+)?(?:millier|milliers|million|millions|milliard|milliards)(?:\s+de)?",
        normalized,
    ):
        return True
    if re.fullmatch(
        r"(?:habitant|habitants|hab|hbts)(?:\s*/\s*(?:km2|km²))?",
        normalized,
    ):
        return True
    return False


def _ascii(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text).casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _period_token(text: str) -> str | None:
    value = _compact(text).strip("()[]:;")
    normalized = _ascii(value).replace("é", "e")
    # Les tableaux budgetaires et conjoncturels expriment souvent une periode
    # cumulee sous la forme ``12 mois-25`` ou ``3 mois-2025``. Le suffixe
    # ``(en`` appartient alors au libelle d'unite de la ligne suivante et non
    # a la periode elle-meme.
    cumulative = re.fullmatch(
        r"(?:\d{1,2}\s+)?mois[-_/ ]?(?:19|20)?\d{2,4}(?:\s*\(?en)?",
        normalized,
        flags=re.I,
    )
    if cumulative:
        return re.sub(r"\(?en$", "", value, flags=re.I).strip()
    # Les en-tetes PDF separent frequemment ``T4-`` et ``2025`` en deux mots.
    # Le trimestre nu doit etre reconnu pour que `_header_periods` puisse les
    # reunir en une seule periode.
    quarter_base = normalized.rstrip("-_/ ")
    if re.fullmatch(r"[tq][1-4]", quarter_base, flags=re.I):
        return quarter_base.upper()
    if _YEAR_RE.fullmatch(normalized):
        return normalized
    # Les classeurs officiels suffixent souvent les années avec le statut du
    # compte (CD, CP, provisoire...). La période reste l'année ; le suffixe est
    # conservé comme qualificatif visible dans la cellule.
    if re.fullmatch(r"(?:19|20)\d{2}\s*[a-z]{1,12}", normalized, flags=re.I):
        return value
    if _QUARTER_RE.fullmatch(normalized) or _MONTH_RE.fullmatch(normalized):
        return value
    return None


def is_period_label(text: str) -> bool:
    """API publique légère utilisée par les importeurs PDF et tableur."""
    return _period_token(text) is not None


def is_temporal_header_row(row: Sequence[str]) -> bool:
    """Distingue un véritable en-tête temporel d'une ligne numérique.

    Une ligne comptable peut contenir par hasard des valeurs telles que 2000
    et 2080. Deux nombres ressemblant à des années ne suffisent donc pas : les
    périodes doivent constituer la majorité de la zone numérique de la ligne
    et les années explicites doivent progresser de gauche à droite.
    """
    non_empty = [(_compact(cell), index) for index, cell in enumerate(row) if _compact(cell)]
    periods = [
        (_period_token(cell), index)
        for cell, index in non_empty
        if _period_token(cell)
    ]
    if len(periods) < 2:
        return False
    # Autorise au plus deux cellules descriptives avant la série (indicateur et
    # unité), mais exige que la partie restante soit majoritairement temporelle.
    if len(periods) / max(1, len(non_empty) - 2) < 0.75:
        return False
    years = []
    for period, _index in periods:
        match = re.search(r"\b((?:19|20)\d{2})\b", _ascii(period or ""))
        if match:
            years.append(int(match.group(1)))
    if len(years) >= 2 and any(right < left for left, right in zip(years, years[1:])):
        return False
    return True


def normalize_numeric_value(value: str) -> str:
    """Retourne une forme numerique stable, sans modifier la valeur affichee."""
    cleaned = _FOOTNOTE_RE.sub("", _compact(value))
    cleaned = cleaned.replace("−", "-").replace("\u202f", " ").replace("\u00a0", " ")
    match = _NUMERIC_RE.search(cleaned)
    if not match:
        return ""
    numeric = match.group(0).replace("%", "").replace(" ", "")
    # Les tableaux francophones utilisent la virgule decimale. Le point reste
    # conserve lorsqu'il est deja le separateur decimal.
    return numeric.replace(",", ".")


@dataclass(frozen=True)
class TableFact:
    page: int
    table_id: str
    row_id: str
    row_index: int
    indicator: str
    period: str
    value: str
    normalized_value: str
    unit: str = ""
    domain: str = ""
    territory: str = ""
    measure_type: str = ""
    qualifier: str = ""
    source_method: str = "pdf_layout"

    def to_dict(self) -> dict:
        return asdict(self)


def is_plausible_table_fact(fact: dict | TableFact) -> bool:
    """Ecarte les cellules manifestement decalees sans juger le fond metier.

    La valeur doit correspondre a une cellule numerique complete, l'indicateur
    a un libelle de ligne raisonnable et l'unite ne doit pas contenir un morceau
    d'une autre colonne. Ce filtre est volontairement conservateur et ne depend
    d'aucune question particuliere.
    """
    data = fact.to_dict() if isinstance(fact, TableFact) else fact
    value = _FOOTNOTE_RE.sub("", _compact(data.get("value", "")))
    if not value or not _STRICT_NUMERIC_VALUE_RE.fullmatch(value):
        return False
    if not normalize_numeric_value(value):
        return False

    indicator = _compact(data.get("indicator", ""))
    if not indicator or len(indicator) > 160:
        return False
    # Un nombre, un poids ou un code de nomenclature pris pour le libelle de
    # ligne signale presque toujours un decalage horizontal du tableau. Un
    # indicateur exploitable doit porter un minimum de semantique textuelle ;
    # les sigles courts comme PIB, IPC ou M1 restent naturellement acceptes.
    if not any(char.isalpha() for char in indicator):
        return False
    if len(re.findall(r"\d+(?:[.,]\d+)?", indicator)) >= 3:
        return False

    period = _compact(data.get("period", ""))
    if not _period_token(period):
        return False
    years = [int(year) for year in re.findall(r"\b(?:19|20)\d{2}\b", period)]
    if any(year > datetime.now().year + 30 for year in years):
        return False

    unit = _compact(data.get("unit", ""))
    if len(unit) > 48:
        return False
    if len(re.findall(r"\d+(?:[.,]\d+)?", unit)) >= 2:
        return False
    if unit and not _is_plausible_unit(unit):
        return False
    return True


def is_plausible_table_row(facts: Iterable[dict | TableFact]) -> bool:
    """Valide la cohérence structurelle d'une ligne avant son indexation.

    Une ligne décalée peut contenir des cellules individuellement numériques
    mais associer deux indicateurs, pages ou unités différents. Dans ce cas, la
    ligne entière est écartée : le texte de la page reste disponible dans le
    corpus, mais aucune relation indicateur-période-valeur douteuse n'est créée.
    """
    rows = [fact.to_dict() if isinstance(fact, TableFact) else dict(fact) for fact in facts]
    if not rows or any(not is_plausible_table_fact(fact) for fact in rows):
        return False

    def distinct(field: str) -> set[str]:
        return {
            _ascii(_compact(fact.get(field, "")))
            for fact in rows if _compact(fact.get(field, ""))
        }

    if len(distinct("indicator")) != 1:
        return False
    if len({int(fact.get("page") or 0) for fact in rows}) != 1:
        return False
    if len(distinct("unit")) > 1 or len(distinct("territory")) > 1:
        return False
    if len(distinct("measure_type")) > 1:
        return False

    values_by_period: dict[str, set[str]] = defaultdict(set)
    for fact in rows:
        period = _ascii(_compact(fact.get("period", "")))
        value = str(
            fact.get("normalized_value")
            or normalize_numeric_value(str(fact.get("value", "")))
        )
        values_by_period[period].add(value)
    return all(len(values) == 1 for values in values_by_period.values())


def rotated_words_to_layout(
    words: Iterable[dict],
    *,
    page_width: float,
    page_height: float,
    clockwise: bool = False,
) -> list[dict]:
    """Projette des mots tournes de 90 degres dans un repere horizontal.

    Certains PDF publient un tableau paysage comme un objet tourne sur une page
    portrait. Les mots restent alors verticaux pour pdfplumber et les colonnes
    deviennent impossibles a reconstruire. La transformation ne modifie pas le
    texte ; elle remet uniquement les boites dans le repere logique du tableau.
    """
    transformed = []
    for source in words:
        if source.get("upright", True):
            continue
        word = dict(source)
        if clockwise:
            word["x0"] = float(source["top"])
            word["x1"] = float(source["bottom"])
            word["top"] = float(page_width) - float(source["x1"])
            word["bottom"] = float(page_width) - float(source["x0"])
        else:
            word["x0"] = float(page_height) - float(source["bottom"])
            word["x1"] = float(page_height) - float(source["top"])
            word["top"] = float(source["x0"])
            word["bottom"] = float(source["x1"])
        transformed.append(word)
    return transformed


def infer_table_context(text: str, *, source_hint: str = "") -> dict[str, str]:
    """Infere les facettes communes d'une page sans interpreter ses valeurs."""
    normalized = _ascii(_compact(text))
    hint = _ascii(source_hint)
    unit = ""
    if (
        re.search(r"\bmilliards?\s+(?:de\s+)?(?:francs?\s+)?(?:fcfa|xof)\b", normalized)
        or (re.search(r"\bmilliards?\b", normalized) and re.search(r"\b(?:fcfa|xof)\b", normalized))
    ):
        unit = "milliards de FCFA"
    elif (
        re.search(r"\bmillions?\s+(?:de\s+)?(?:francs?\s+)?(?:fcfa|xof)\b", normalized)
        or (re.search(r"\bmillions?\b", normalized) and re.search(r"\b(?:fcfa|xof)\b", normalized))
    ):
        unit = "millions de FCFA"
    elif re.search(r"\bindice\w*\b", normalized) and re.search(
        r"\b(?:prix|production|chiffre d affaires|harmonise)\b", normalized,
    ):
        unit = "indice"
    elif "%" in text or "pourcentage" in normalized:
        unit = "%"
    elif re.search(r"\bindice\w*\b", normalized):
        unit = "indice"

    facets = []
    if re.search(r"\b(?:cvs|corrige\w* des variations saisonnier)\b", normalized):
        facets.append("CVS")
    if re.search(r"\b(?:volume|prix constant)\b", normalized):
        facets.append("volume")
    if re.search(r"\b(?:valeur|nominal|prix courant)\b", normalized):
        facets.append("nominal")
    if "indice" in normalized and "prix" in normalized:
        facets.append("indice de prix")
    chained = re.search(r"prix (?:de reference |)\s*((?:19|20)\d{2})", normalized)
    if "chaine" in normalized and chained:
        facets.append(f"prix chaines de {chained.group(1)}")

    territory = ""
    if re.search(r"\bcote d ivoire\b", normalized) or "anstat" in hint:
        territory = "Cote d'Ivoire"
    return {
        "unit": unit,
        "territory": territory,
        "measure_type": " | ".join(dict.fromkeys(facets)),
    }


def _group_words_by_line(
    words: Iterable[dict], *, tolerance: float = 2.6,
) -> list[list[dict]]:
    ordered = sorted(words, key=lambda word: (float(word["top"]), float(word["x0"])))
    lines: list[list[dict]] = []
    line_tops: list[float] = []
    for word in ordered:
        top = float(word["top"])
        if not lines or abs(top - line_tops[-1]) > tolerance:
            lines.append([word])
            line_tops.append(top)
        else:
            lines[-1].append(word)
            line_tops[-1] = sum(float(item["top"]) for item in lines[-1]) / len(lines[-1])
    for line in lines:
        line.sort(key=lambda word: float(word["x0"]))
    return lines


def _word_center(word: dict) -> float:
    return (float(word["x0"]) + float(word["x1"])) / 2.0


def _line_text(words: Sequence[dict]) -> str:
    return _compact(" ".join(str(word.get("text", "")) for word in words))


def _header_periods(line: Sequence[dict]) -> list[tuple[str, float]]:
    """Lit les périodes, y compris les libellés répartis sur deux mots.

    pdfplumber renvoie fréquemment ``Décembre`` et ``2025`` comme deux mots
    distincts. Sans regroupement, ils seraient pris pour deux colonnes alors
    qu'ils décrivent ensemble une seule observation mensuelle.
    """
    ordered = sorted(line, key=lambda word: float(word["x0"]))
    periods: list[tuple[str, float]] = []
    index = 0
    while index < len(ordered):
        word = ordered[index]
        text = str(word.get("text", ""))
        # ``12`` et ``mois-25`` sont frequemment deux mots PDF distincts.
        # On essaie d'abord leur combinaison, uniquement s'ils sont voisins,
        # avant d'interpreter chaque mot separement.
        if index + 1 < len(ordered):
            next_word = ordered[index + 1]
            horizontal_gap = float(next_word["x0"]) - float(word["x1"])
            if horizontal_gap <= 24.0:
                combined = f"{text} {next_word.get('text', '')}"
                combined_period = _period_token(combined)
                if combined_period:
                    periods.append((
                        combined_period,
                        (_word_center(word) + _word_center(next_word)) / 2.0,
                    ))
                    index += 2
                    continue
        period = _period_token(text)
        if not period:
            index += 1
            continue

        normalized = _ascii(_compact(text).strip("()[]:;"))
        is_unqualified_month = bool(_MONTH_RE.fullmatch(normalized)) and not re.search(
            r"\d", normalized,
        )
        is_unqualified_quarter = bool(
            re.fullmatch(r"[tq][1-4](?:[-_/])?", normalized, flags=re.I)
        ) and not re.search(r"(?:19|20)\d{2}", normalized)
        if (is_unqualified_month or is_unqualified_quarter) and index + 1 < len(ordered):
            next_word = ordered[index + 1]
            next_text = _compact(str(next_word.get("text", ""))).strip("()[]:;")
            horizontal_gap = float(next_word["x0"]) - float(word["x1"])
            if _YEAR_RE.fullmatch(_ascii(next_text)) and horizontal_gap <= 24.0:
                periods.append((
                    f"{period} {next_text}",
                    (_word_center(word) + _word_center(next_word)) / 2.0,
                ))
                index += 2
                continue

        periods.append((period, _word_center(word)))
        index += 1
    return periods


def _looks_like_domain(text: str) -> bool:
    compact = _compact(text)
    letters = [char for char in compact if char.isalpha()]
    if not letters or len(compact) > 90:
        return False
    uppercase_ratio = sum(char.isupper() for char in letters) / len(letters)
    return uppercase_ratio >= 0.75


def _split_indicator_unit(
    left_words: Sequence[dict], *, unit_left: float,
) -> tuple[str, str]:
    indicator_words = [word for word in left_words if _word_center(word) < unit_left]
    unit_words = [word for word in left_words if _word_center(word) >= unit_left]
    indicator = _line_text(indicator_words)
    unit = _line_text(unit_words)
    return indicator, unit


def _clean_unit(unit: str) -> str:
    compact = _compact(unit)
    if not compact:
        return ""
    tokens = _ascii(compact).split()
    if tokens and all(re.fullmatch(r"[-+]?\d+(?:[.,]\d+)?", token) for token in tokens):
        return ""
    return compact if _is_plausible_unit(compact) else ""


def _clean_indicator(text: str) -> str:
    compact = _compact(text).strip("-|:;")
    return re.sub(r"\s+([,;)])", r"\1", compact)


def _cell_value(words: Sequence[dict]) -> tuple[str, str]:
    raw = _line_text(words)
    if not raw or not any(char.isdigit() for char in raw):
        return "", ""
    qualifier_match = _FOOTNOTE_RE.search(raw)
    qualifier = qualifier_match.group(0).strip() if qualifier_match else ""
    value = _FOOTNOTE_RE.sub("", raw).strip()
    return value, qualifier


def extract_layout_table_facts(
    words: Iterable[dict],
    *,
    page_number: int,
    min_periods: int = 2,
    max_facts: int = 500,
    default_unit: str = "",
    territory: str = "",
    measure_type: str = "",
) -> list[dict]:
    """Reconstruit les tableaux de series a partir des positions des mots.

    Le mecanisme fonctionne aussi avec des lignes clairsemees : une valeur est
    associee a la colonne visuelle sous laquelle elle se trouve, pas a son rang
    dans le texte extrait. Cela evite les decalages d'annees responsables des
    erreurs sur les tableaux ANStat.
    """
    lines = _group_words_by_line(words)
    header_candidates: list[tuple[int, list[tuple[str, float]], list[dict]]] = []
    for index, line in enumerate(lines):
        periods = _header_periods(line)
        if len(periods) >= min_periods:
            header_candidates.append((index, periods, line))
    if not header_candidates:
        return []

    facts: list[TableFact] = []
    for table_number, (header_index, periods, header_line) in enumerate(header_candidates, start=1):
        # Une seconde ligne de periodes avant le pied de page marque un nouveau
        # tableau. Elle fixe donc la borne de traitement du tableau courant.
        later_headers = [item[0] for item in header_candidates if item[0] > header_index]
        stop_index = min(later_headers) if later_headers else len(lines)

        period_centers = [center for _period, center in periods]
        if len(period_centers) < 2:
            continue
        gaps = [right - left for left, right in zip(period_centers, period_centers[1:])]
        first_gap = gaps[0]
        last_gap = gaps[-1]
        bounds = [period_centers[0] - first_gap / 2]
        bounds.extend(
            (left + right) / 2
            for left, right in zip(period_centers, period_centers[1:])
        )
        bounds.append(period_centers[-1] + last_gap / 2)

        unit_headers = [
            word for word in header_line
            if _ascii(str(word.get("text", ""))).strip(".:;") in {"unite", "unit"}
        ]
        unit_left = (
            float(unit_headers[0]["x0"]) - 16.0
            if unit_headers else bounds[0] - max(35.0, first_gap)
        )
        # Une page en deux colonnes peut juxtaposer un tableau d'indices et un
        # tableau financier. Le contexte global de la page attribuerait alors
        # l'unite du second au premier. On infere donc le contexte dans la seule
        # bande horizontale du tableau courant, autour de son en-tete.
        # La ligne logique peut contenir simultanement du texte de la colonne
        # voisine. On remonte depuis le libelle le plus proche de la premiere
        # valeur et on s'arrete au premier grand espace horizontal : cela garde
        # ``Indice Global`` dans un tableau gauche, tout en excluant le texte
        # narratif voisin d'un tableau droit.
        header_labels = sorted(
            (
                word for word in header_line
                if _word_center(word) < bounds[0]
            ),
            key=lambda word: float(word["x0"]),
        )
        label_cluster: list[dict] = []
        if header_labels:
            label_cluster = [header_labels[-1]]
            for previous in reversed(header_labels[:-1]):
                gap = float(label_cluster[-1]["x0"]) - float(previous["x1"])
                if gap > 20.0:
                    break
                label_cluster.append(previous)
        local_left = (
            min(float(word["x0"]) for word in label_cluster) - 10.0
            if label_cluster else unit_left - 25.0
        )
        # Le cadrage horizontal n'est necessaire que lorsqu'un autre bloc de
        # texte partage effectivement la ligne de l'en-tete (mise en page en
        # colonnes). Sur un tableau pleine largeur, il pourrait sinon couper le
        # debut d'un indicateur long ou une unite legerement decalee.
        excluded_header_words = [
            word for word in header_line
            if word not in label_cluster and _word_center(word) < local_left
        ]
        side_by_side_layout = bool(
            label_cluster
            and any(
                _ascii(str(word.get("text", ""))).strip(".:;-")
                not in {
                    "indicateur", "indicateurs", "cle", "rubrique", "libelle",
                    "unite", "unit",
                }
                for word in excluded_header_words
            )
        )
        local_right = bounds[-1] + last_gap
        header_top = sum(float(word["top"]) for word in header_line) / len(header_line)
        local_words = [
            word
            for line in lines
            for word in line
            if (
                local_left <= _word_center(word) <= local_right
                and header_top - 45.0 <= float(word["top"]) <= header_top + 45.0
            )
        ]
        local_context = infer_table_context(_line_text(local_words))
        table_default_unit = local_context["unit"] or default_unit
        local_measure = local_context["measure_type"]
        if local_measure:
            table_measure_type = " | ".join(dict.fromkeys(
                facet.strip()
                for facet in f"{measure_type} | {local_measure}".split("|")
                if facet.strip()
            ))
        elif local_context["unit"]:
            # Une unite locale identifie un tableau distinct sur une page en
            # colonnes. On n'herite alors pas des facettes du tableau voisin.
            table_measure_type = ""
        else:
            table_measure_type = measure_type
        if not unit_headers and local_context["unit"]:
            # L'unite est portee par l'en-tete du tableau (par exemple
            # ``12 mois-25 (en milliards de FCFA)``), pas par une colonne
            # intermediaire. Toute la zone avant la premiere valeur appartient
            # donc au libelle de l'indicateur.
            unit_left = bounds[0]
        table_id = f"page-{page_number}-period-table-{table_number}"
        domain = ""
        pending_indicator: list[str] = []
        pending_unit: list[str] = []
        current_row: dict | None = None
        row_index = 0

        def finalize_current() -> None:
            nonlocal current_row, row_index
            if not current_row:
                return
            raw_unit = _compact(" ".join(current_row["unit_parts"]))
            unit = _clean_unit(raw_unit)
            indicator_parts = list(current_row["indicator_parts"])
            # Si le texte placé dans la colonne d'unité n'en est pas une, il
            # s'agit le plus souvent de la suite horizontale du libellé. Le
            # rattacher à l'indicateur répare des cas généraux comme « Burkina
            # Faso » et « formation brute de capital fixe » sans dictionnaire
            # propre à un document.
            if raw_unit and not unit:
                if _ascii(raw_unit).strip("'’ -") in _AMBIGUOUS_LABEL_FRAGMENTS:
                    current_row = None
                    return
                indicator_parts.append(raw_unit)
            indicator = _clean_indicator(" ".join(indicator_parts))
            unit = unit or table_default_unit
            # Une ligne en capitales peut etre un agrégat valide (PIB,
            # PRIMAIRE, AGRICULTURE VIVRIERE). Les domaines sans valeur sont
            # deja identifies plus bas ; on ne rejette donc jamais une ligne
            # numerique uniquement a cause de sa casse.
            if not indicator:
                current_row = None
                return
            row_index += 1
            row_id = f"{table_id}-row-{row_index}"
            for period, cell_words in zip(periods, current_row["cells"]):
                value, qualifier = _cell_value(cell_words)
                normalized_value = normalize_numeric_value(value)
                if not value or not normalized_value:
                    continue
                fact = TableFact(
                    page=page_number,
                    table_id=table_id,
                    row_id=row_id,
                    row_index=row_index,
                    indicator=indicator,
                    period=period[0],
                    value=value,
                    normalized_value=normalized_value,
                    unit=unit,
                    domain=domain,
                    territory=territory,
                    measure_type=table_measure_type,
                    qualifier=qualifier,
                )
                if is_plausible_table_fact(fact):
                    facts.append(fact)
                if len(facts) >= max_facts:
                    break
            current_row = None

        for source_line in lines[header_index + 1:stop_index]:
            # Les pages en deux colonnes ont des lignes partageant presque la
            # meme ordonnee. Sans ce cadrage, un libelle du tableau de gauche
            # pouvait etre concatene a une ligne financiere du tableau droit.
            line = (
                [
                    word for word in source_line
                    if local_left <= _word_center(word) <= local_right
                ]
                if side_by_side_layout else list(source_line)
            )
            if not line:
                continue
            text = _line_text(line)
            normalized_text = _ascii(text)
            line_top = min(float(word["top"]) for word in line)
            close_to_current = bool(
                current_row
                and line_top - float(current_row["top"]) <= 18.0
            )
            full_line_tokens = set(re.findall(r"[a-z]+", normalized_text))
            if full_line_tokens and full_line_tokens <= {
                "chiffres", "cles", "de", "la", "cote", "d", "ivoire",
            }:
                continue
            if normalized_text.startswith(("source :", "source:", "periodicite", "tableau des")):
                finalize_current()
                break
            # Les brochures ANStat placent le titre du document verticalement
            # dans la marge gauche. Un vrai domaine de tableau est centre dans
            # la zone de donnees, contrairement a ces fragments decoratifs.
            domain_words = [word for word in line if float(word["x0"]) >= 150.0]
            domain_text = _line_text(domain_words)
            normalized_domain = _ascii(domain_text)
            domain_tokens = set(re.findall(r"[a-z]+", normalized_domain))
            page_title_fragment = bool(domain_tokens) and domain_tokens <= {
                "chiffres", "cles", "de", "la", "cote", "d", "ivoire",
            }
            if (
                domain_text
                and _looks_like_domain(domain_text)
                and not any(char.isdigit() for char in domain_text)
                and not page_title_fragment
                and not close_to_current
                and not any(
                    marker in normalized_domain
                    for marker in ("chiffres cles", "cote d'ivoire", "indicateur cle")
                )
            ):
                finalize_current()
                domain = _clean_indicator(domain_text)
                pending_indicator = []
                pending_unit = []
                continue

            cells: list[list[dict]] = [[] for _ in periods]
            left_words: list[dict] = []
            for word in line:
                center = _word_center(word)
                assigned = False
                for cell_index in range(len(periods)):
                    if bounds[cell_index] <= center < bounds[cell_index + 1]:
                        cells[cell_index].append(word)
                        assigned = True
                        break
                if not assigned and center < bounds[0]:
                    left_words.append(word)
            has_values = any(_cell_value(cell)[0] for cell in cells)
            indicator_part, unit_part = _split_indicator_unit(
                left_words, unit_left=unit_left,
            )

            if has_values:
                finalize_current()
                current_row = {
                    "top": min(float(word["top"]) for word in line),
                    "indicator_parts": pending_indicator + ([indicator_part] if indicator_part else []),
                    "unit_parts": pending_unit + ([unit_part] if unit_part else []),
                    "cells": cells,
                }
                pending_indicator = []
                pending_unit = []
                continue

            if not indicator_part and not unit_part:
                continue
            if _looks_like_domain(text) and not unit_part and not close_to_current:
                finalize_current()
                domain = _clean_indicator(text)
                pending_indicator = []
                pending_unit = []
                continue

            vertical_gap = line_top - float(current_row["top"]) if current_row else 999.0
            continuation_hint = bool(
                indicator_part
                and (
                    indicator_part[:1].islower()
                    or indicator_part.startswith(("(", "-", "/"))
                )
            )
            if current_row and (
                vertical_gap <= 10.5
                or (continuation_hint and vertical_gap <= 18.0)
            ):
                if indicator_part:
                    current_row["indicator_parts"].append(indicator_part)
                if unit_part:
                    current_row["unit_parts"].append(unit_part)
                current_row["top"] = line_top
            else:
                finalize_current()
                if indicator_part:
                    pending_indicator.append(indicator_part)
                if unit_part:
                    pending_unit.append(unit_part)
        finalize_current()

        if len(facts) >= max_facts:
            break

    # Certains moteurs peuvent detecter deux fois le meme en-tete. La cle
    # factuelle rend la sortie stable et evite les embeddings dupliques.
    unique: dict[tuple, dict] = {}
    for fact in facts:
        payload = fact.to_dict()
        key = (
            fact.page, _ascii(fact.indicator), _ascii(fact.period),
            fact.normalized_value, _ascii(fact.unit),
        )
        unique.setdefault(key, payload)
    return list(unique.values())[:max_facts]


def extract_matrix_table_facts(
    rows: Sequence[Sequence[str]],
    *,
    page_number: int = 0,
    table_number: int = 1,
    default_unit: str = "",
    territory: str = "",
    measure_type: str = "",
) -> list[dict]:
    """Convertit un tableau HTML/PDF deja structure en faits atomiques."""
    if len(rows) < 2:
        return []
    header_index = -1
    periods: list[tuple[int, str]] = []
    for index, row in enumerate(rows):
        if not is_temporal_header_row([str(cell or "") for cell in row]):
            continue
        candidate = [
            (column, period)
            for column, cell in enumerate(row)
            if (period := _period_token(str(cell or "")))
        ]
        if len(candidate) >= 2:
            header_index = index
            periods = candidate
            break
    if header_index < 0:
        return []

    header = [str(cell or "") for cell in rows[header_index]]
    unit_column = next(
        (
            index for index, cell in enumerate(header)
            if _ascii(cell).strip(".:;") in {"unite", "unit"}
        ),
        None,
    )
    first_period_column = min(column for column, _period in periods)
    facts: list[dict] = []
    table_id = f"page-{page_number}-matrix-table-{table_number}"
    domain = ""
    row_counter = 0
    for row in rows[header_index + 1:]:
        cells = [str(cell or "").strip() for cell in row]
        if not any(cells):
            continue
        label_cells = [
            cell for column, cell in enumerate(cells[:first_period_column])
            if cell and column != unit_column
        ]
        indicator = _clean_indicator(label_cells[-1]) if label_cells else ""
        qualifier = _compact(" > ".join(label_cells[:-1]))
        unit = _clean_unit(cells[unit_column]) if unit_column is not None and unit_column < len(cells) else ""
        unit = unit or default_unit
        if _looks_like_domain(indicator) and not any(
            column < len(cells) and normalize_numeric_value(cells[column])
            for column, _period in periods
        ):
            domain = indicator
            continue
        if not indicator:
            continue
        row_counter += 1
        row_id = f"{table_id}-row-{row_counter}"
        row_facts: list[dict] = []
        for column, period in periods:
            if column >= len(cells):
                continue
            value = _FOOTNOTE_RE.sub("", cells[column]).strip()
            normalized_value = normalize_numeric_value(value)
            if not normalized_value:
                continue
            row_facts.append(TableFact(
                page=page_number,
                table_id=table_id,
                row_id=row_id,
                row_index=row_counter,
                indicator=indicator,
                period=period,
                value=value,
                normalized_value=normalized_value,
                unit=unit,
                domain=domain,
                territory=territory,
                measure_type=measure_type,
                qualifier=qualifier,
                source_method="structured_matrix",
            ).to_dict())
        if row_facts and is_plausible_table_row(row_facts):
            facts.extend(row_facts)
    return facts


def deduplicate_table_facts(facts: Iterable[dict]) -> list[dict]:
    unique: dict[tuple, dict] = {}
    for fact in facts:
        # Defense en profondeur : les donnees brutes historiques peuvent avoir
        # ete produites par une ancienne version de l'extracteur. Le nettoyage
        # les revalide ainsi avant tout chunking ou embedding.
        if not is_plausible_table_fact(fact):
            continue
        key = (
            int(fact.get("page") or 0),
            _ascii(fact.get("indicator", "")),
            _ascii(fact.get("period", "")),
            fact.get("normalized_value") or normalize_numeric_value(fact.get("value", "")),
            _ascii(fact.get("unit", "")),
            _ascii(fact.get("territory", "")),
            _ascii(fact.get("measure_type", "")),
        )
        if key[1] and key[2] and key[3]:
            unique.setdefault(key, dict(fact))
    return list(unique.values())


def group_facts_by_row(facts: Iterable[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for fact in facts:
        row_id = str(fact.get("row_id") or fact.get("table_id") or "table-row")
        grouped[row_id].append(dict(fact))
    return dict(grouped)
