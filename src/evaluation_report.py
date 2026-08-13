"""Export PDF lisible des evaluations question par question."""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def _number(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _as_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return []
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _font_paths() -> tuple[Path | None, Path | None]:
    candidates = [
        (
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/arialbd.ttf"),
        ),
        (
            Path("C:/Windows/Fonts/segoeui.ttf"),
            Path("C:/Windows/Fonts/seguisb.ttf"),
        ),
    ]
    for regular, bold in candidates:
        if regular.exists() and bold.exists():
            return regular, bold
    return None, None


def export_evaluation_pdf(
    results: Iterable[dict[str, Any]],
    summary: dict[str, Any],
    output_path: Path,
    *,
    title: str = "Évaluation RAG - Banque de 100 questions",
    subtitle: str = "Questions, réponses générées, sources retrouvées et temps de réponse",
) -> Path:
    """Genere un rapport PDF A4 complet et retourne son chemin."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        HRFlowable,
        PageBreak,
        Paragraph,
        KeepTogether,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    rows = list(results)
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    regular_path, bold_path = _font_paths()
    regular_font = "Helvetica"
    bold_font = "Helvetica-Bold"
    if regular_path and bold_path:
        regular_font = "RAGRegular"
        bold_font = "RAGBold"
        if regular_font not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(regular_font, str(regular_path)))
        if bold_font not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(bold_font, str(bold_path)))

    navy = colors.HexColor("#10233F")
    blue = colors.HexColor("#155EEF")
    pale_blue = colors.HexColor("#EEF4FF")
    pale_green = colors.HexColor("#ECFDF3")
    pale_red = colors.HexColor("#FEF3F2")
    ink = colors.HexColor("#172B4D")
    muted = colors.HexColor("#5E6C84")
    border = colors.HexColor("#D8E0EA")

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "RAGTitle",
        parent=styles["Title"],
        fontName=bold_font,
        fontSize=22,
        leading=27,
        textColor=navy,
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    subtitle_style = ParagraphStyle(
        "RAGSubtitle",
        parent=styles["Normal"],
        fontName=regular_font,
        fontSize=10.5,
        leading=15,
        textColor=muted,
        alignment=TA_CENTER,
        spaceAfter=18,
    )
    section_style = ParagraphStyle(
        "RAGSection",
        parent=styles["Heading2"],
        fontName=bold_font,
        fontSize=14,
        leading=18,
        textColor=navy,
        spaceBefore=8,
        spaceAfter=9,
    )
    question_style = ParagraphStyle(
        "RAGQuestion",
        parent=styles["Heading3"],
        fontName=bold_font,
        fontSize=12,
        leading=16,
        textColor=navy,
        spaceAfter=6,
        keepWithNext=True,
    )
    label_style = ParagraphStyle(
        "RAGLabel",
        parent=styles["Normal"],
        fontName=bold_font,
        fontSize=9.5,
        leading=13,
        textColor=ink,
        spaceBefore=5,
        spaceAfter=2,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "RAGBody",
        parent=styles["BodyText"],
        fontName=regular_font,
        fontSize=9.5,
        leading=14,
        textColor=ink,
        spaceAfter=5,
        allowWidows=0,
        allowOrphans=0,
    )
    small_style = ParagraphStyle(
        "RAGSmall",
        parent=body_style,
        fontSize=8.3,
        leading=11.5,
        textColor=muted,
        wordWrap="CJK",
    )
    metric_style = ParagraphStyle(
        "RAGMetric",
        parent=body_style,
        fontSize=8.8,
        leading=12,
        textColor=ink,
    )

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=18 * mm,
        bottomMargin=17 * mm,
        title=title,
        author="RAG ENSEA",
        subject="Évaluation automatisée du golden set Banque 100",
    )

    def escaped(value: Any) -> str:
        return html.escape(str("" if value is None else value)).replace("\n", "<br/>")

    def footer(canvas, doc) -> None:
        canvas.saveState()
        canvas.setStrokeColor(border)
        canvas.line(16 * mm, 13 * mm, A4[0] - 16 * mm, 13 * mm)
        canvas.setFont(regular_font, 7.5)
        canvas.setFillColor(muted)
        canvas.drawString(16 * mm, 8.5 * mm, "RAG ENSEA - Évaluation Banque 100")
        canvas.drawRightString(
            A4[0] - 16 * mm,
            8.5 * mm,
            f"Page {doc.page}",
        )
        canvas.restoreState()

    story: list[Any] = [
        Spacer(1, 16 * mm),
        Paragraph(escaped(title), title_style),
        Paragraph(escaped(subtitle), subtitle_style),
        Paragraph(
            f"Rapport généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}",
            subtitle_style,
        ),
        Spacer(1, 4 * mm),
        Paragraph("Résultats globaux", section_style),
    ]

    metric_labels = {
        "questions_total": "Questions prévues",
        "questions_completed": "Questions traitées",
        "errors": "Erreurs d'exécution",
        "answerable_questions": "Questions répondables",
        "unanswerable_questions": "Questions hors corpus",
        "automated_pass_rate": "Taux de succès automatisé",
        "retrieval_hit_rate_at_5": "Hit rate document @5",
        "page_hit_rate_at_5": "Hit rate page PDF @5",
        "citation_source_hit_rate": "Sources attendues citées",
        "answer_supported_by_citation_rate": "Réponses étayées par une citation",
        "reference_relation_score_mean": "Concordance globale avec la référence",
        "lead_reference_relation_score_mean": "Concordance de la réponse principale",
        "abstention_accuracy": "Précision d'abstention",
        "route_accuracy": "Précision du routage",
        "route_effective_accuracy": "Couverture effective du routage",
        "mean_total_seconds": "Temps total moyen (s)",
        "p50_total_seconds": "Temps total median (s)",
        "p95_total_seconds": "Temps total p95 (s)",
        "max_total_seconds": "Temps total maximal (s)",
        "mean_retrieval_seconds": "Retrieval moyen (s)",
        "mean_generation_seconds": "Génération moyenne (s)",
    }
    metric_rows = [[
        Paragraph("<b>Indicateur</b>", body_style),
        Paragraph("<b>Valeur</b>", metric_style),
    ]]
    for key, label in metric_labels.items():
        if key not in summary:
            continue
        value = summary[key]
        if "rate" in key or key.endswith("accuracy"):
            value = f"{100 * float(value):.1f} %"
        elif "seconds" in key:
            value = _number(value, 2)
        metric_rows.append([
            Paragraph(escaped(label), body_style),
            Paragraph(escaped(value), metric_style),
        ])
    summary_table = Table(metric_rows, colWidths=[118 * mm, 42 * mm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), pale_blue),
        ("FONTNAME", (0, 0), (-1, -1), regular_font),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.35, border),
        ("BOX", (0, 0), (-1, -1), 0.65, border),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([
        summary_table,
        Spacer(1, 6 * mm),
        Paragraph(
            "Les indicateurs automatiques facilitent le diagnostic, mais ne remplacent "
            "pas une vérification humaine des réponses complexes.",
            small_style,
        ),
        PageBreak(),
        Paragraph("Résultats question par question", section_style),
    ])

    for position, row in enumerate(rows, start=1):
        question_story: list[Any] = []
        number = row.get("question_number") or position
        status = "IN-BASE" if row.get("should_answer") else "OUT-OF-BASE"
        passed = bool(row.get("automated_pass")) and not row.get("error")
        badge_fill = pale_green if passed else pale_red
        heading_table = Table(
            [[
                Paragraph(
                    f"{escaped(number)}. {escaped(row.get('question'))}",
                    question_style,
                ),
                Paragraph(
                    f"<b>{status}</b><br/>{'SUCCÈS' if passed else 'À VÉRIFIER'}",
                    small_style,
                ),
            ]],
            colWidths=[132 * mm, 28 * mm],
        )
        heading_table.setStyle(TableStyle([
            ("BACKGROUND", (1, 0), (1, 0), badge_fill),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (1, 0), (1, 0), "CENTER"),
            ("LEFTPADDING", (0, 0), (0, 0), 0),
            ("RIGHTPADDING", (0, 0), (0, 0), 8),
            ("LEFTPADDING", (1, 0), (1, 0), 4),
            ("RIGHTPADDING", (1, 0), (1, 0), 4),
            ("TOPPADDING", (1, 0), (1, 0), 4),
            ("BOTTOMPADDING", (1, 0), (1, 0), 4),
            ("BOX", (1, 0), (1, 0), 0.5, border),
        ]))
        question_story.append(heading_table)

        meta = (
            f"Domaine : {escaped(row.get('domain_label') or row.get('category'))} | "
            f"Route : {escaped(row.get('route'))} "
            f"(attendue : {escaped(row.get('expected_route') or '-')}) | "
            f"Temps total : {_number(row.get('total_seconds'), 2)} s | "
            f"Retrieval : {_number(row.get('retrieval_seconds'), 2)} s | "
            f"Génération : {_number(row.get('generation_seconds'), 2)} s"
        )
        question_story.append(Paragraph(meta, metric_style))

        if row.get("error"):
            question_story.extend([
                Paragraph("Erreur", label_style),
                Paragraph(escaped(row.get("error")), body_style),
            ])
        else:
            question_story.extend([
                Paragraph("Réponse de référence", label_style),
                Paragraph(escaped(row.get("reference_answer")), body_style),
                Paragraph("Réponse du RAG", label_style),
                Paragraph(escaped(row.get("generated_answer")), body_style),
            ])

        question_story.extend([
            Paragraph("Source attendue", label_style),
            Paragraph(escaped(row.get("source_locator") or "Aucune source attendue"), small_style),
        ])
        cited_sources = _as_list(row.get("citations"))
        retrieved_sources = _as_list(row.get("retrieved_sources"))
        if not cited_sources and not retrieved_sources:
            question_story.append(Paragraph("Sources trouvées", label_style))
            question_story.append(Paragraph("Aucune source trouvée ou citée.", small_style))
        if cited_sources:
            question_story.append(Paragraph("Sources citées dans la réponse", label_style))
            for source_index, source in enumerate(cited_sources, start=1):
                title_text = source.get("title") or source.get("section") or "Source"
                url_text = source.get("url") or ""
                source_text = (
                    f"<b>[{source_index}] {escaped(title_text)}</b><br/>"
                    f"{escaped(url_text)}"
                )
                question_story.append(Paragraph(source_text, small_style))
        if retrieved_sources:
            question_story.append(Paragraph("Sources retrouvées par le retriever", label_style))
            for source_index, source in enumerate(retrieved_sources, start=1):
                title_text = source.get("title") or source.get("section") or "Source"
                url_text = source.get("url") or ""
                score = source.get("rerank_score")
                score_text = f" - score {_number(score, 3)}" if score is not None else ""
                source_text = (
                    f"<b>[{source_index}] {escaped(title_text)}{score_text}</b><br/>"
                    f"{escaped(url_text)}"
                )
                question_story.append(Paragraph(source_text, small_style))

        question_story.extend([
            Spacer(1, 2.5 * mm),
            HRFlowable(width="100%", thickness=0.6, color=border),
            Spacer(1, 4 * mm),
        ])
        story.append(KeepTogether(question_story))

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return output_path
