"""Routage lexical explicable entre les corpus du RAG ENSEA.

Le routeur distingue les connaissances institutionnelles stables, les données
économiques et les activités étudiantes temporelles. Il combine des familles de
signaux plutôt que d'exiger une formulation littérale unique. Les alias des
pages Facebook viennent de la configuration afin que l'ajout d'une nouvelle
source ne nécessite pas une règle codée en dur.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

Route = Literal["institutional", "economic", "facebook", "both", "all"]


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    confidence: float
    reason: str


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


_EXPLICIT_FACEBOOK_PHRASES = {
    "page facebook", "post facebook", "publication facebook",
    "ensea junior services", "junior services", "ejs",
    "ensea data science club", "data science club", "edsc",
    "depeche ecostat", "ecostat",
    "association des eleves", "aes",
    "division des ingenieurs statisticiens economistes", "dise",
    "division des analystes statisticiens", "das",
    "club de debat et de rhetorique", "cdr",
    "concours ensea d eloquence", "concours d eloquence",
    "stat hack", "journee de l analyste", "celeste diner gala",
    "semaine culturelle", "phare de l entrepreneuriat",
}

_OFFICIAL_RECENT_PHRASES = {
    "derniere actualite", "dernieres actualites", "publication recente",
    "publications recentes", "recemment publie", "proces verbal du jury",
    "appel a communications", "dernier resultat", "derniers resultats",
}

# Informations pour lesquelles le site et les documents officiels doivent
# rester prioritaires, même si la question contient le mot « concours » ou une
# année. Cela évite de confondre les admissions avec la vie associative.
_INSTITUTIONAL_PRIORITY_PHRASES = {
    "concours d entree", "concours entree", "concours d admission",
    "procedure d admission", "procedures d admission", "conditions d admission",
    "resultats d admission", "liste des admis", "calendrier academique",
    "accreditation", "programme de formation", "filiere ise", "filiere as",
    "filiere its", "contact de l ensea", "adresse de l ensea",
}

# Activités typiquement documentées par les pages des clubs et associations.
# Ces termes seuls ne suffisent pas : ils sont combinés avec un événement, un
# résultat ou une date afin de ne pas détourner les inventaires institutionnels.
_STUDENT_ACTIVITY_PATTERNS = (
    r"\beloquen[a-z]*\b", r"\borateur[a-z]*\b", r"\bgolden speaker\b",
    r"\bclub[a-z]*\b", r"\bassociation[a-z]*\b", r"\bdivision[a-z]*\b",
    r"\bbureau[a-z]*\b", r"\bmandat[a-z]*\b", r"\bequipe[a-z]*\b",
    r"\bgala[a-z]*\b", r"\bhackathon[a-z]*\b", r"\bstat hack\b",
    r"\bjournee culturelle\b", r"\bvie associative\b",
)

_EVENT_PATTERNS = (
    r"\bconcour[a-z]*\b", r"\bcompetition[a-z]*\b", r"\btournoi[a-z]*\b",
    r"\bfinale[a-z]*\b", r"\bdemi finale[a-z]*\b", r"\bevenement[a-z]*\b",
    r"\belection[a-z]*\b", r"\bassemblee generale\b", r"\bceremonie[a-z]*\b",
    r"\bcampagne[a-z]*\b", r"\bconference[a-z]*\b",
)

_OUTCOME_PATTERNS = (
    r"\bremport[a-z]*\b", r"\bgagn[a-z]*\b", r"\bvainqueur[a-z]*\b",
    r"\blaureat[a-z]*\b", r"\bsacre[a-z]*\b", r"\bfinaliste[a-z]*\b",
    r"\bclassement[a-z]*\b", r"\bresultat[a-z]*\b", r"\belu[a-z]*\b",
    r"\bnouvelle equipe\b", r"\bnouveau bureau\b", r"\ba la tete\b",
    r"\bpresident[a-z]*\b", r"\bdirecteur general\b", r"\bdg\b",
)

_RECENCY_PATTERNS = (
    r"\b20\d{2}\b", r"\bcette annee\b", r"\bannee (?:scolaire|academique)\b",
    r"\bdernier[a-z]*\b", r"\brecent[a-z]*\b", r"\bnouveau[a-z]*\b",
    r"\bprochain[a-z]*\b", r"\bactuel[a-z]*\b", r"\bedition[a-z]*\b",
)

# Domaines structurels pour lesquels le corpus institutionnel est la source
# naturelle. Cette liste conserve une voie rapide sans transformer chaque
# question inconnue en exclusion silencieuse du corpus Facebook.
_INSTITUTIONAL_STABLE_PATTERNS = (
    r"\badmission[a-z]*\b", r"\badmis[a-z]*\b", r"\baccreditation[a-z]*\b",
    r"\bformation[a-z]*\b", r"\bfiliere[a-z]*\b",
    r"\blaboratoire[a-z]*\b", r"\bpartenaire[a-z]*\b",
    r"\bprogramme academique\b", r"\bproduction scientifique\b",
    r"\btravaux de recherche\b", r"\bcontact[a-z]*\b",
    r"\badresse[a-z]*\b", r"\blocalisation[a-z]*\b",
    r"\bhistoire de l ensea\b", r"\bmission[a-z]* de l ensea\b",
    r"\borganigramme[a-z]*\b", r"\bdirecteur general de l ensea\b",
    r"\bdg de l ensea\b", r"\bdirection de l ensea\b",
)

_MIXED_STUDENT_ORGANIZATION_PHRASES = {
    "clubs et associations", "clubs et association",
    "club et associations", "club et association",
    "associations et clubs", "associations et club",
    "association et clubs", "association et club",
    "organisations etudiantes", "organisation etudiante",
    "vie associative",
}

_ECONOMIC_PHRASES = {
    "anstat", "agence nationale de la statistique",
    "statistiques de cote d ivoire", "cote d ivoire en chiffres",
    "comptes nationaux", "compte national", "produit interieur brut", "pib",
    "inflation", "prix a la consommation", "ihpc",
    # Les utilisateurs ne connaissent pas nécessairement le nom officiel des
    # indices ANSTAT. Ces concepts couvrent aussi leurs formulations usuelles.
    "indice de production industrielle", "production industrielle",
    "activite industrielle", "secteur industriel", "production manufacturiere",
    "production extractive", "volume de production industrielle", "ipi",
    "indice de prix de production", "prix de production industrielle", "ippi",
    "recensement general", "rgph", "population de la cote d ivoire",
    "repertoire national des entreprises", "centrale de bilans",
    "tableau de bord socio economique", "tbse",
    "chiffre d affaires dans le commerce", "finances publiques",
    "cadrage macroeconomique", "indicateurs macroeconomiques",
}

# Familles d'indicateurs statistiques, indépendantes de la formulation exacte
# d'une question. Elles couvrent les domaines présents dans les publications
# ANStat et économiques sans faire de termes génériques tels que « taux » ou
# « combien » des signaux suffisants à eux seuls.
_ECONOMIC_DOMAIN_PATTERNS = (
    # Démographie et conditions de vie
    r"\bdemograph[a-z]*\b", r"\bdensite de (?:la )?population\b",
    r"\bpopulation de la region\b",
    r"\brapport de masculinite\b", r"\bpopulation (?:masculine|feminine)\b",
    r"\bpersonnes? agees? de \d+\b", r"\bmortalite (?:infantile|maternelle|infanto)\b",
    # Santé, énergie et accès aux services essentiels
    r"\betablissement sanitaire\b", r"\bcentre hospitalier universitaire\b",
    r"\bchu\b", r"\bmedecins? (?:etaient |sont )?recenses?\b",
    r"\bsages femmes? (?:etaient |sont )?recensees?\b",
    r"\bcouverture vaccinale\b", r"\bdeces dus? au paludisme\b",
    r"\belectrification\b", r"\bacces a l electricite\b",
    r"\bdesserte (?:des )?menages? en electricite\b",
    # Activité économique, entreprises et finances publiques
    r"\bsecteur (?:primaire|secondaire|tertiaire)\b",
    r"\bexportations?\b", r"\bimportations?\b", r"\binvestissements?\b",
    r"\bconsommation finale\b", r"\bchiffre d affaires\b",
    r"\bentreprises? formelles?\b", r"\bproportion de pme\b",
    r"\bexcedent brut d exploitation\b", r"\bebe\b",
    r"\brecettes? fiscales?\b", r"\bdeficit budgetaire\b", r"\btaux directeur\b",
    r"\bdette publique\b", r"\bbalance des paiements\b",
    r"\btelecommunications?\b",
    # Éducation, emploi, pauvreté et gouvernance statistique
    r"\beleves? (?:etaient |sont )?inscrits? au (?:primaire|secondaire)\b",
    r"\btaux (?:net )?de scolarisation\b", r"\btaux d achevement\b",
    r"\btaux d abandon scolaire\b", r"\btaux d alphabetisation\b",
    r"\bindice de parite filles garcons\b", r"\benseignants? hommes?\b",
    r"\btaux de chomage\b", r"\btaux d emploi\b", r"\btaux de pauvrete\b",
    r"\bindice de gini\b", r"\bconnexion internet a domicile\b",
    r"\bpart de la consommation\b", r"\bindice de perception de la corruption\b",
)

_NATIONAL_STATISTICAL_SCOPE_PATTERNS = (
    r"\bcote d ivoire\b", r"\bivoirien[a-z]*\b", r"\bnational[a-z]*\b",
)

_STATISTICAL_MEASURE_PATTERNS = (
    r"\btaux\b", r"\bindice\b", r"\bpart\b", r"\bproportion\b",
    r"\bcroissance\b", r"\bvariation\b", r"\bmontant\b", r"\bvaleur\b",
    r"\bniveau\b", r"\bpopulation\b", r"\bcombien\b", r"\bsolde\b",
)


def _contains_phrase(normalized_query: str, phrase: str) -> bool:
    return re.search(rf"(?:^| ){re.escape(phrase)}(?: |$)", normalized_query) is not None


def _matches_any(normalized_query: str, patterns: tuple[str, ...]) -> list[str]:
    """Retourne les motifs présents pour rendre la décision observable."""
    return [pattern for pattern in patterns if re.search(pattern, normalized_query)]


def _configured_page_matches(
    normalized_query: str,
    page_aliases: dict[str, list[str]] | None,
) -> list[str]:
    matches: list[str] = []
    for page_key, aliases in (page_aliases or {}).items():
        if any(
            (alias_normalized := _normalize(alias))
            and _contains_phrase(normalized_query, alias_normalized)
            for alias in aliases
        ):
            matches.append(page_key)
    return matches


def resolve_facebook_page_scope(
    question: str,
    page_aliases: dict[str, list[str]] | None,
) -> str | None:
    """Retourne la page Facebook explicitement nommee, sans regle metier codee.

    Les alias vivent dans ``config.yaml``. Si plusieurs pages correspondent,
    la recherche reste volontairement non filtree afin de ne pas masquer une
    source pertinente sur la base d'une ambiguite.
    """
    normalized = _normalize(question)
    matches: list[tuple[int, str]] = []
    for page_key, aliases in (page_aliases or {}).items():
        for alias in aliases:
            normalized_alias = _normalize(alias)
            if normalized_alias and _contains_phrase(normalized, normalized_alias):
                matches.append((len(normalized_alias), page_key))
                break
    matched_pages = {page_key for _length, page_key in matches}
    if len(matched_pages) != 1:
        return None
    return max(matches)[1]


def route_question(
    question: str,
    page_aliases: dict[str, list[str]] | None = None,
) -> RouteDecision:
    normalized = _normalize(question)
    economic = sorted(
        phrase for phrase in _ECONOMIC_PHRASES
        if _contains_phrase(normalized, phrase)
    )
    if economic:
        return RouteDecision(
            "economic", 0.98,
            f"source économique explicite : {economic[0]}",
        )

    economic_domains = _matches_any(normalized, _ECONOMIC_DOMAIN_PATTERNS)
    national_scope = _matches_any(
        normalized, _NATIONAL_STATISTICAL_SCOPE_PATTERNS,
    )
    statistical_measure = _matches_any(
        normalized, _STATISTICAL_MEASURE_PATTERNS,
    )
    if economic_domains or (national_scope and statistical_measure):
        evidence = (
            economic_domains[0]
            if economic_domains
            else f"{national_scope[0]} + {statistical_measure[0]}"
        )
        return RouteDecision(
            "economic", 0.94,
            f"indicateur statistique national : {evidence}",
        )

    student_organizations = sorted(
        phrase for phrase in _MIXED_STUDENT_ORGANIZATION_PHRASES
        if _contains_phrase(normalized, phrase)
    )
    if student_organizations:
        return RouteDecision(
            "institutional", 0.98,
            "inventaire d'appartenance : priorité à la liste institutionnelle",
        )

    institutional = sorted(
        phrase for phrase in _INSTITUTIONAL_PRIORITY_PHRASES
        if _contains_phrase(normalized, phrase)
    )
    if institutional:
        return RouteDecision(
            "institutional", 0.98,
            f"information institutionnelle stable : {institutional[0]}",
        )

    configured_pages = _configured_page_matches(normalized, page_aliases)
    explicit = sorted(
        phrase for phrase in _EXPLICIT_FACEBOOK_PHRASES
        if _contains_phrase(normalized, phrase)
    )
    if configured_pages or explicit:
        signal = configured_pages[0] if configured_pages else explicit[0]
        return RouteDecision(
            "facebook", 0.99,
            f"structure ou source étudiante explicite : {signal}",
        )

    stable = _matches_any(normalized, _INSTITUTIONAL_STABLE_PATTERNS)
    if stable:
        return RouteDecision(
            "institutional", 0.94,
            "connaissance institutionnelle stable ou administrative",
        )

    activity = _matches_any(normalized, _STUDENT_ACTIVITY_PATTERNS)
    events = _matches_any(normalized, _EVENT_PATTERNS)
    outcomes = _matches_any(normalized, _OUTCOME_PATTERNS)
    recency = _matches_any(normalized, _RECENCY_PATTERNS)

    # Une activité étudiante associée à un événement, un résultat ou une date
    # est beaucoup plus susceptible de vivre dans les publications sociales.
    if activity and (events or outcomes or recency):
        evidence = ", ".join(
            label for label, values in (
                ("activité étudiante", activity),
                ("événement", events),
                ("résultat", outcomes),
                ("repère temporel", recency),
            ) if values
        )
        return RouteDecision(
            "facebook", 0.96,
            f"actualité de la vie étudiante : {evidence}",
        )

    # Pour une actualité ENSEA non rattachée à une structure précise, interroger
    # les deux corpus protège le rappel sans abandonner la source officielle.
    if events and (outcomes or recency):
        return RouteDecision(
            "both", 0.88,
            "événement ENSEA temporel : recherche institutionnelle et sociale",
        )

    recent = sorted(
        phrase for phrase in _OFFICIAL_RECENT_PHRASES
        if _contains_phrase(normalized, phrase)
    )
    if recent:
        return RouteDecision("both", 0.90, f"actualité officielle potentiellement récente : {recent[0]}")

    # Un routeur lexical ne peut pas anticiper toutes les formulations. Par
    # défaut, aucun corpus ne doit être supprimé avant même le retrieval. La
    # route « all » interroge donc institutionnel, économique et Facebook ;
    # « both » reste réservé au couple institutionnel + Facebook des actualités
    # ENSEA. Les routes exclusives ci-dessus exigent un signal explicite.
    return RouteDecision(
        "all", 0.55,
        "route ambiguë : recherche de rappel sur tous les corpus disponibles",
    )
