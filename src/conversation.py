"""Contextualisation légère des questions de suivi, sans appel LLM supplémentaire."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    normalized = "".join(
        character for character in normalized
        if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", normalized).strip()


def is_contextual_followup(question: str) -> bool:
    """Détecte une demande qui dépend manifestement du tour précédent."""
    normalized = _normalize(question)
    markers = (
        r"\b(cette liste|la liste|liste complete|liste exhaustive|exhaustif|exhaustive)\b",
        r"\b(ta reponse|votre reponse|reponse precedente|message precedent)\b",
        r"\b(ces derniers|ces dernieres|ceux ci|celles ci|le premier|le deuxieme|"
        r"la premiere|la deuxieme|ce dernier|cette derniere)\b",
        # Groupe nominal demonstratif sans referent explicite : "cette equipe",
        # "ce projet", "ces personnes". Les expressions temporelles peuvent
        # en revanche etre autonomes ("cette annee", "ce mois").
        r"\b(?:ce|cet|cette|ces)\s+(?!(?:annee|mois|semaine|jour|matin|soir|"
        r"nuit|heure)\b)[a-z0-9-]+\b",
        r"\b(plus de details|davantage de details|precise|developpe|continue)\b",
        r"^(et|mais|donc|alors)\b",
        r"\b(qu en est il|est ce complet|est elle complete|est il complet)\b",
        r"\b(il|elle|ils|elles|lui|leur|celui|celle|ceux|celles|cela|ca|"
        r"son|sa|ses|leurs|precedent|precedente|meme|autre)\b",
        # Pronoms compléments élidés : « Qui l'a précédé ? », « Où l'a-t-il
        # étudié ? ». On exige un auxiliaire après l'apostrophe pour ne pas
        # confondre avec un nom comme « l'ENSEA » ou « l'école ».
        r"\bl['’](?:a|avait|est|etait|sera|aurait)\b",
        r"\b(he|she|it|they|him|her|them|his|their|former|latter|"
        r"previous answer|previous message)\b",
        r"^(and|but|so|then|what about)\b",
        r"^(non|pas exactement|je parlais|je voulais dire|plutot)\b",
        r"^(qui|quoi|quand|ou|pourquoi|comment|combien|lequel|laquelle|"
        r"lesquels|lesquelles)\s*[?!.]*$",
    )
    if any(re.search(marker, normalized) for marker in markers):
        return True
    # Un mot relationnel ne suffit pas à faire d'une question une relance.
    # « Quels directeurs se sont succédé à l'ENSEA ? » est autonome, alors que
    # « Qui a précédé ? » ou « Les prédécesseurs ? » omettent bien leur sujet.
    tokens = re.findall(r"[a-z0-9]+", normalized)
    if len(tokens) <= 7 and re.fullmatch(
        r"(?:qui|who)(?:\s+(?:a|avait|has|had|did))?\s+"
        r"(?:preced|succed|remplac)\w*\s*[?!.]*",
        normalized,
    ):
        return True
    if len(tokens) <= 10 and re.match(
        r"^(?:les?|la|un|une|des)\s+", normalized,
    ) and re.search(
        r"\b(?:predecesseur|successeur|preced|succed|remplac)\w*\b",
        normalized,
    ):
        return True
    return len(tokens) <= 7 and bool(re.match(
        r"^(et|mais|pour|en|les?|la|un|une|and|but|so|then)\b", normalized,
    ))


def _assistant_anchor(history: list[dict[str, Any]] | None) -> str:
    """Extrait un court ancrage lexical de la dernière réponse.

    L'ancrage sert uniquement à retrouver les bons documents. Il n'est jamais
    transmis comme preuve et ses anciens numéros de citation sont supprimés.
    Garder la première phrase suffit généralement à conserver l'entité ou le
    sujet résolu au tour précédent sans polluer la recherche avec toute la
    réponse.
    """
    for message in reversed(history or []):
        if message.get("role") != "assistant":
            continue
        content = " ".join(str(message.get("content", "")).split())
        if not content:
            continue
        normalized = _normalize(content)
        if normalized.startswith((
            "je ne trouve pas cette information",
            "bonjour comment puis je vous aider",
            "avec plaisir",
            "d accord je vous ecoute",
            "a bientot",
            "je suis l assistant documentaire",
        )):
            continue
        content = re.sub(r"\[[0-9\s,;]+\]", "", content)
        first_sentences = re.split(r"(?<=[.!?])\s+", content)
        anchor = " ".join(first_sentences[:2]).strip()
        return anchor[:360].rstrip()
    return ""


def contextualize_question(
    question: str,
    history: list[dict[str, Any]] | None,
) -> tuple[str, bool]:
    """Ajoute la dernière question utilisateur uniquement pour un vrai suivi."""
    if not history or not is_contextual_followup(question):
        return question, False

    previous_questions = [
        str(message.get("content", "")).strip()
        for message in reversed(history)
        if message.get("role") == "user" and str(message.get("content", "")).strip()
    ]
    substantive_questions = [
        candidate for candidate in previous_questions
        if direct_conversation_reply(candidate) is None
    ]
    previous_user_question = next(
        (
            candidate for candidate in substantive_questions
            if not is_contextual_followup(candidate)
        ),
        substantive_questions[0] if substantive_questions else "",
    )
    if not previous_user_question:
        return question, False
    correction = re.match(
        r"^(non|pas exactement|je parlais|je voulais dire|plutot)\b",
        _normalize(question),
    )
    anchor = "" if correction else _assistant_anchor(history)
    parts = [f"Question précédente : {previous_user_question}"]
    if anchor:
        parts.append(f"Référence résolue au tour précédent : {anchor}")
    parts.append(f"Question de suivi : {question}")
    return "\n".join(parts), True


def retrieval_subject_text(question: str) -> str:
    """Retire l'habillage technique d'une requête conversationnelle.

    ``contextualize_question`` ajoute des libellés destinés au routeur et au
    retrieval. Ces libellés ne font pas partie de la demande utilisateur : le
    mot « précédente » ne doit notamment jamais être interprété comme une
    relation de succession entre les entités recherchées. La référence issue
    de l'ancienne réponse est également exclue car elle aide au rappel global,
    mais ne constitue pas un critère documentaire à prouver.
    """
    kept: list[str] = []
    for raw_line in str(question).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        normalized = _normalize(line)
        if normalized.startswith("reference resolue au tour precedent :"):
            continue
        if normalized.startswith((
            "question precedente :", "question de suivi :",
        )):
            _label, _separator, content = line.partition(":")
            if content.strip():
                kept.append(content.strip())
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def direct_conversation_reply(message: str) -> str | None:
    """Répond aux tours sociaux simples sans lancer le pipeline documentaire.

    Les motifs sont volontairement des énoncés complets et courts : une phrase
    comme « Merci, et qui l'a précédé ? » doit continuer vers le RAG.
    """
    normalized = re.sub(
        r"\s+", " ", re.sub(r"[-'’]+", " ", _normalize(message))
    ).strip(" .!?…")
    if re.fullmatch(r"(?:bonjour|bonsoir|salut|coucou|hello|hi|hey)(?: a toi)?", normalized):
        if normalized in {"hello", "hi", "hey"}:
            return "Hello! How can I help you with ENSEA and its documentary resources?"
        return "Bonjour ! Comment puis-je vous aider au sujet de l’ENSEA ou des documents disponibles ?"
    if re.fullmatch(
        r"(?:merci|merci beaucoup|je te remercie|je vous remercie|thanks|thank you|many thanks)",
        normalized,
    ):
        return "Avec plaisir. Nous pouvons poursuivre sur ce sujet ou en explorer un autre."
    if re.fullmatch(
        r"(?:d accord|ok|okay|compris|tres bien|parfait|ca marche|entendu|got it|understood)",
        normalized,
    ):
        return "D’accord. Je vous écoute pour la suite."
    if re.fullmatch(r"(?:au revoir|a bientot|bonne journee|bye|goodbye|see you)", normalized):
        return "À bientôt !"
    if re.fullmatch(
        r"(?:qui es tu|qui etes vous|what are you|who are you|que peux tu faire|"
        r"que pouvez vous faire|comment peux tu m aider|comment pouvez vous m aider)",
        normalized,
    ):
        return (
            "Je suis l’assistant documentaire de l’ENSEA. Je peux rechercher des "
            "informations dans les sources institutionnelles, les rapports économiques, "
            "les PDF ajoutés et les publications des clubs, poursuivre une conversation "
            "et fournir les sources utilisées."
        )
    return None


def is_exhaustive_request(question: str) -> bool:
    """Réserve plus de résultats aux demandes explicitement exhaustives."""
    normalized = _normalize(question)
    return bool(re.search(
        r"\b(exhaustif|exhaustive|exhaustifs|exhaustives|tous|toutes|"
        r"liste complete|au complet|sans exception)\b",
        normalized,
    ))


def exhaustive_category_tokens(question: str) -> list[str]:
    """Repère le nom générique de la collection demandée.

    La fonction ne connaît aucun domaine particulier. Elle reconnaît des
    formulations interrogatives courantes puis applique la même normalisation
    plurielle légère que l'index lexical.
    """
    normalized = _normalize(question)
    primary_question = re.split(
        r"\bquestion\s+de\s+suivi\s*:", normalized, maxsplit=1,
    )[0]
    primary_question = re.sub(
        r"^\s*question\s+precedente\s*:\s*", "", primary_question,
    )
    # Les marqueurs de discours sont fréquents dans une vraie conversation :
    # « quels sont donc les... », « quelles sont exactement les... ». Ils ne
    # portent pas le sujet documentaire et ne doivent pas devenir l'ancre de
    # recherche. La liste reste volontairement grammaticale et indépendante
    # du domaine (directeurs, clubs, accréditations, etc.).
    discourse_markers = (
        r"(?:donc|alors|exactement|precisement|finalement|reellement|"
        r"concretement|maintenant|aujourd hui)"
    )
    patterns = (
        rf"\b(?:quels|quelles)\s+sont(?:\s+{discourse_markers}){{0,2}}\s+"
        r"(?:les|des)\s+([a-z0-9-]+)",
        r"\b(?:quels|quelles)\s+([a-z0-9-]+)",
        r"\b(?:liste|inventaire|noms?)(?:\s+[a-z0-9-]+){0,2}\s+"
        r"(?:des|du|de|d['’]?)\s*([a-z0-9-]+)",
        r"\b(?:tous|toutes)\s+(?:les|des)\s+([a-z0-9-]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, primary_question)
        if not match:
            continue
        token = match.group(1)
        if len(token) > 5 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        return [token] if token else []
    return []


def is_collection_request(question: str) -> bool:
    """Détecte une question qui demande naturellement plusieurs entités.

    Une formulation comme « quels sont les laboratoires ? » implique déjà
    une couverture de liste, même sans employer le mot « exhaustif ».
    """
    return bool(exhaustive_category_tokens(question))


def is_membership_request(question: str) -> bool:
    """Détecte les inventaires qui exigent une relation d'appartenance prouvée."""
    normalized = _normalize(question)
    membership_entities = (
        r"clubs?|associations?|organisations?|divisions?|laboratoires?|"
        r"equipes?|membres?|filiales?|departements?|partenaires?"
    )
    relation_markers = (
        r"\b(?:de|du|des|a|au|aux|appartenant a|rattache(?:e|es|s)? a|"
        r"present(?:e|es|s)? a|associe(?:e|es|s)? a)\b"
    )
    return bool(
        re.search(rf"\b(?:{membership_entities})\b", normalized)
        and re.search(relation_markers, normalized)
    )


_EVIDENCE_STOPWORDS = {
    "a", "au", "aux", "avec", "ce", "ces", "cette", "comment", "d",
    "dans", "de", "des", "du", "elle", "en", "ensea", "est", "et",
    "il", "ils", "l", "la", "le", "les", "leur", "leurs", "on", "ou",
    "par", "pour", "qu", "que", "quel", "quelle", "quelles", "quels",
    "qui", "sa", "se", "ses", "son", "sont", "sur", "un", "une",
}


def evidence_token_stems(text: str) -> set[str]:
    """Produit des racines lexicales légères pour localiser une preuve.

    Elles servent uniquement à choisir, dans un passage déjà récupéré, les
    phrases les plus proches de la question lorsque le budget du prompt impose
    une compression. Elles ne remplacent ni BM25 ni l'embedding.
    """
    stems: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", _normalize(text)):
        if token in _EVIDENCE_STOPWORDS or len(token) < 3:
            continue
        if len(token) > 5 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        stems.add(token[:7] if len(token) > 7 else token)
    return stems


def evidence_sentences(text: str, query: str) -> list[str]:
    """Classe les phrases d'un passage par proximité avec la question."""
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?;:])\s+|[\r\n]+", text)
        if sentence.strip()
    ]
    query_stems = evidence_token_stems(query)
    if not sentences or not query_stems:
        return sentences
    scored = []
    for position, sentence in enumerate(sentences):
        overlap = query_stems & evidence_token_stems(sentence)
        # Réunir plusieurs notions de la question est plus probant qu'une
        # occurrence isolée d'un terme générique.
        score = len(overlap) * len(overlap)
        scored.append((score, -position, sentence))
    scored.sort(reverse=True)
    return [sentence for _score, _position, sentence in scored]


_ROLE_ALIAS_GROUPS = (
    ("directeur general", "directrice generale", "dg", "d g"),
    ("vice president", "vice presidente"),
    ("president", "presidente"),
    ("secretaire general", "secretaire generale"),
    ("secretaire",),
    ("tresorier", "tresoriere"),
    ("directeur", "directrice"),
    ("responsable",),
    ("coordinateur", "coordinatrice"),
    ("chef",),
    ("gerant", "gerante"),
    ("fondateur", "fondatrice"),
)


def requested_role_aliases(question: str) -> tuple[str, ...]:
    """Retourne le titre de fonction explicitement demande, du plus precis au plus large."""
    normalized = _normalize(question)
    for aliases in _ROLE_ALIAS_GROUPS:
        if any(
            re.search(rf"(?:^|\s){re.escape(alias)}(?:\s|$)", normalized)
            for alias in aliases
        ):
            return aliases
    return ()


def filter_unproven_ocr_roles(context: str, question: str) -> str:
    """Ecarte l'OCR qui ne contient pas la fonction demandee.

    Un fragment visuel reste utile pour un nom de liste ou un libelle. Il ne
    peut toutefois relier une personne a une fonction que si cette fonction
    apparait dans le meme fragment OCR. La legende textuelle reste intacte.
    """
    aliases = requested_role_aliases(question)
    if not aliases:
        return context
    kept: list[str] = []
    inside_ocr = False
    kept_ocr_lines: list[str] = []
    for line in context.splitlines():
        normalized_line = _normalize(line)
        if re.match(r"^#{1,6}\s*texte (?:detecte|ocr)", normalized_line):
            inside_ocr = True
            continue
        if not inside_ocr:
            kept.append(line)
            continue
        if any(
            re.search(rf"(?:^|\s){re.escape(alias)}(?:\s|$)", normalized_line)
            for alias in aliases
        ):
            kept_ocr_lines.append(line)
    if kept_ocr_lines:
        kept.extend(["", "## OCR contenant la fonction demandee", *kept_ocr_lines])
    return "\n".join(kept).strip()


def generation_history(
    question: str,
    history: list[dict[str, Any]] | None,
    *,
    contextualized: bool,
) -> list[dict[str, Any]]:
    """Évite qu'une ancienne réponse incomplète ancre une nouvelle liste.

    Une liste autonome est reconstruite sans historique. Pour une relance
    elliptique (« la liste complète ? »), les questions utilisateur suffisent
    à retrouver le sujet ; les anciennes réponses ne sont pas des preuves.
    """
    messages = history or []
    # Une question autonome ouvre un nouveau sujet documentaire, meme dans la
    # meme conversation. L'ancien echange ne doit jamais devenir une source
    # implicite de noms, dates ou organisations pour ce nouveau sujet.
    if not contextualized:
        return []
    # Une relance ne concerne que le sujet autonome le plus recent. Sans cette
    # frontiere, une conversation EJS -> CDR -> "et son president ?" pourrait
    # encore exposer au modele les faits du bloc EJS.
    topic_start = 0
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") != "user":
            continue
        content = str(message.get("content", "")).strip()
        if (
            content
            and direct_conversation_reply(content) is None
            and not is_contextual_followup(content)
        ):
            topic_start = index
            break
    messages = messages[topic_start:]
    if is_exhaustive_request(question) or is_collection_request(question):
        return [message for message in messages if message.get("role") == "user"]
    return messages


def compact_history(
    history: list[dict[str, Any]] | None,
    *,
    max_messages: int = 4,
    max_characters_per_message: int = 420,
) -> list[dict[str, str]]:
    """Conserve quelques tours récents bornés pour le prompt de génération."""
    compacted: list[dict[str, str]] = []
    for message in (history or [])[-max_messages:]:
        role = str(message.get("role", ""))
        content = " ".join(str(message.get("content", "")).split())
        if role not in {"user", "assistant"} or not content:
            continue
        if role == "assistant":
            # Les anciens numéros [n] appartiennent à un ancien contexte et ne
            # doivent jamais être réutilisés comme citations du nouveau tour.
            content = re.sub(r"\[[0-9\s,;]+\]", "", content)
            content = re.sub(r"\s+([,.;:!?])", r"\1", " ".join(content.split()))
        if len(content) > max_characters_per_message:
            content = content[:max_characters_per_message].rstrip() + "…"
        compacted.append({"role": role, "content": content})
    return compacted


def update_memory_summary(
    previous_summary: str,
    user_message: str,
    assistant_message: str,
    *,
    max_characters: int = 3600,
) -> str:
    """Maintient une mémoire conversationnelle compacte sans second appel LLM.

    Cette mémoire n'est jamais une source documentaire. Elle sert uniquement à
    résoudre les pronoms et les relances quand les messages détaillés les plus
    anciens ne sont plus injectés dans le prompt.
    """
    clean_answer = re.sub(r"\[[0-9\s,;]+\]", "", assistant_message)
    clean_answer = " ".join(clean_answer.split())[:900]
    clean_question = " ".join(user_message.split())[:400]
    entry = f"Utilisateur : {clean_question}\nAssistant : {clean_answer}"
    blocks = [block.strip() for block in previous_summary.split("\n\n") if block.strip()]
    blocks.append(entry)
    while blocks and len("\n\n".join(blocks)) > max_characters:
        blocks.pop(0)
    return "\n\n".join(blocks)
