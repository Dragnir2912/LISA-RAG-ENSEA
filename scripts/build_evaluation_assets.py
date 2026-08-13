"""Génère le golden set enrichi et les notebooks d'évaluation spécialisés.

Les notebooks sont des artefacts JSON générés afin de garder leur contenu
reproductible et facile à relire dans les revues de code.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


BASE_METADATA = [
    ("formation_initiale", "medium"), ("formation_initiale", "easy"),
    ("formation_initiale", "easy"), ("admission", "medium"),
    ("admission", "medium"), ("financement", "medium"),
    ("gouvernance", "hard"), ("orientation", "medium"),
    ("recherche", "hard"), ("recherche", "medium"),
    ("cooperation", "medium"), ("contact", "easy"),
    ("formation_initiale", "easy"), ("formation_initiale", "easy"),
    ("formation_initiale", "easy"), ("gouvernance", "medium"),
    ("personnel", "medium"), ("admission", "medium"),
    ("admission", "medium"), ("juridique", "easy"),
]


def example(
    question: str,
    answer: str,
    urls: list[str],
    category: str,
    keywords: list[str],
    difficulty: str = "medium",
    should_answer: bool = True,
) -> dict:
    return {
        "question": question,
        "reference_answer": answer,
        "relevant_urls": urls,
        "category": category,
        "difficulty": difficulty,
        "should_answer": should_answer,
        "answer_keywords": keywords,
    }


ADDITIONAL = [
    example("Quelles associations étudiantes sont présentées par l'ENSEA ?", "Les associations étudiantes sont recensées sur une page dédiée à la vie associative.", ["https://ensea360.ensea.ed.ci/associations-detudiants/"], "vie_etudiante", ["associations", "étudiants"]),
    example("Où puis-je consulter l'équipe de l'ENSEA ?", "La composition de l'équipe est publiée sur la page Notre équipe.", ["https://ensea360.ensea.ed.ci/a-propos/notre-equipe/", "https://ensea.ed.ci/notre-equipe/"], "personnel", ["équipe"], "easy"),
    example("Quelle est l'histoire de l'ENSEA ?", "L'histoire institutionnelle de l'ENSEA est présentée sur sa page Histoire.", ["https://ensea.ed.ci/histoire/", "https://ensea360.ensea.ed.ci/histoire/"], "institution", ["histoire", "ENSEA"]),
    example("Quelle politique qualité l'ENSEA applique-t-elle ?", "L'ENSEA publie sa politique qualité sur une page institutionnelle dédiée.", ["https://ensea.ed.ci/politique-qualite-de-lensea/"], "qualite", ["politique", "qualité"]),
    example("Où trouver les réponses aux questions fréquentes sur l'ENSEA ?", "La FAQ rassemble les réponses aux questions fréquentes.", ["https://ensea360.ensea.ed.ci/faq/", "https://ensea.ed.ci/faq-items/", "https://ensea.ed.ci/frequently-asked-questions/"], "orientation", ["FAQ"], "easy"),
    example("Que propose le Master en Management de l'ENSEA ?", "L'ENSEA présente un Master en Management dans son offre de formation initiale.", ["https://ensea.ed.ci/formation-initiale/master-en-management/"], "formation_initiale", ["Master", "Management"]),
    example("Existe-t-il un master consacré aux infrastructures routières à l'ENSEA ?", "Oui, l'ENSEA propose un Master en Gestion des Infrastructures Routières.", ["https://ensea360.ensea.ed.ci/formation-initiale/master-en-gestion-des-infrastructures-routieres/"], "formation_initiale", ["Master", "infrastructures", "routières"]),
    example("Que couvre la formation continue de l'ENSEA ?", "L'offre de formation continue est décrite sur la page dédiée.", ["https://ensea.ed.ci/formation-continue/", "https://ensea360.ensea.ed.ci/formation-continue"], "formation_continue", ["formation", "continue"]),
    example("Quels certificats de formation continue sont proposés ?", "Les certificats sont présentés dans la rubrique Formation continue.", ["https://ensea.ed.ci/formation-continue/certificats/", "https://ensea.ed.ci/formation-continue/certificats-2/", "https://ensea.ed.ci/formation-continue/"], "formation_continue", ["certificats"]),
    example("Comment est organisée la formation doctorale de l'ENSEA ?", "La formation doctorale possède une présentation et un programme dédiés.", ["https://ensea.ed.ci/formation-doctorale/", "https://ensea360.ensea.ed.ci/formation-doctorale/"], "doctorat", ["formation", "doctorale"]),
    example("Quelles sont les modalités d'admission au doctorat de l'ENSEA ?", "Les modalités sont publiées sur la page Admission Doctorat.", ["https://ensea360.ensea.ed.ci/formation-doctorale/admission/"], "doctorat", ["admission", "doctorat"]),
    example("Quel est le programme de la formation doctorale ?", "Le programme doctoral est détaillé sur une page spécifique.", ["https://ensea.ed.ci/formation-doctorale/programme-formation-doctorale/"], "doctorat", ["programme", "doctorale"]),
    example("Qui enseigne dans la formation doctorale ?", "La liste des enseignants de la formation doctorale est disponible sur le site.", ["https://ensea.ed.ci/formation-doctorale/enseignants-formation-doctorale/"], "personnel", ["enseignants", "doctorale"]),
    example("Où consulter la liste des doctorants de l'ENSEA ?", "Les doctorants sont présentés dans la rubrique Formation doctorale.", ["https://ensea.ed.ci/formation-doctorale/doctorants/"], "doctorat", ["doctorants"]),
    example("L'ENSEA organise-t-elle des séminaires de recherche ?", "Oui, une rubrique est consacrée aux séminaires de recherche.", ["https://ensea360.ensea.ed.ci/formation-continue/seminaires-de-recherche/"], "recherche", ["séminaires", "recherche"]),
    example("Quels organismes sont partenaires de recherche de l'ENSEA ?", "Les partenaires de recherche sont regroupés sur les pages dédiées à la coopération scientifique.", ["https://ensea.ed.ci/cooperation/partenaires-de-recherche/", "https://ensea360.ensea.ed.ci/partenaires-de-recherche/"], "cooperation", ["partenaires", "recherche"]),
    example("Quels projets de coopération l'ENSEA conduit-elle ?", "Les projets de coopération sont présentés dans la rubrique Coopération.", ["https://ensea.ed.ci/cooperation/projets-de-cooperation/", "https://ensea360.ensea.ed.ci/projets-de-cooperation/"], "cooperation", ["projets", "coopération"]),
    example("Quels sont les partenaires institutionnels de l'ENSEA ?", "Une page recense les partenaires institutionnels de l'école.", ["https://ensea.ed.ci/cooperation/partenaires-institutionnels/", "https://ensea360.ensea.ed.ci/partenaires-institutionnels/"], "cooperation", ["partenaires", "institutionnels"]),
    example("Quels sont les partenaires académiques de l'ENSEA ?", "Les partenaires académiques sont répertoriés sur le site de l'ENSEA.", ["https://ensea.ed.ci/listing-partenaires/", "https://ensea360.ensea.ed.ci/partenaires-academiques/"], "cooperation", ["partenaires", "académiques"]),
    example("Avec quelles entreprises l'ENSEA travaille-t-elle ?", "Une page présente les entreprises partenaires de l'ENSEA.", ["https://ensea360.ensea.ed.ci/entreprises-partenaires/", "https://ensea.ed.ci/cooperation/partenaires-entreprises/"], "cooperation", ["entreprises", "partenaires"]),
    example("Où trouver les stages et offres d'emploi diffusés par l'ENSEA ?", "Ils sont publiés dans la rubrique Stages et offres d'emploi.", ["https://ensea360.ensea.ed.ci/formation-initiale/stages-et-offres-demploi/"], "insertion", ["stages", "emploi"]),
    example("Où trouver les anciens sujets de concours de l'ENSEA ?", "Les anciens sujets sont regroupés sur une page de préparation dédiée.", ["https://ensea360.ensea.ed.ci/anciens-sujets/"], "admission", ["anciens", "sujets"]),
    example("Comment contacter le Mastère 3ES ?", "Le site du Mastère 3ES publie une page de contact.", ["https://m3es.ensea.ed.ci/contact/"], "contact", ["contact", "3ES"]),
    example("Quelles sont les conditions d'admission au Mastère 3ES ?", "Les conditions d'admission du Mastère 3ES sont détaillées sur son site.", ["https://m3es.ensea.ed.ci/conditions-dadmission/"], "admission", ["admission", "3ES"]),
    example("Quels bailleurs soutiennent le Mastère 3ES ?", "Les partenaires financiers et bailleurs du Mastère 3ES sont présentés sur une page dédiée.", ["https://m3es.ensea.ed.ci/partenaires-financiers-bailleurs/"], "financement", ["partenaires", "financiers", "bailleurs"]),
    example("L'ENSEA propose-t-elle un master en évaluation des politiques publiques ?", "Oui, une page est consacrée au Master en Evaluation des Politiques Publiques.", ["https://ensea.ed.ci/master-en-evaluation-des-politiques-publiques/"], "formation_initiale", ["Master", "évaluation", "politiques", "publiques"]),
    example("Où sont indiqués les coûts des formations initiales ?", "La page Coûts présente les informations financières relatives à la formation initiale.", ["https://ensea.ed.ci/formation-initiale/couts/"], "financement", ["coûts"]),
    example("Où puis-je consulter les tarifs de l'ENSEA ?", "Une entrée de la FAQ est consacrée aux tarifs.", ["https://ensea.ed.ci/faq-items/quels-sont-les-tarifs/"], "financement", ["tarifs"]),
    example("Comment mobiliser un compte FDFP pour financer une formation ?", "La FAQ explique comment mobiliser un compte FDFP pour le financement.", ["https://ensea.ed.ci/faq-items/comment-puis-je-mobiliser-mon-compte-fdfp-pour-financer-ma-formation/"], "financement", ["FDFP", "financer", "formation"], "hard"),
    example("Quel niveau faut-il pour intégrer la filière AS ?", "Le niveau de recrutement des Analystes Statisticiens est expliqué dans la FAQ.", ["https://ensea.ed.ci/faq-items/quel-est-le-niveau-de-recrutement-pour-integrer-la-filiere-des-analystes-statisticiens/"], "admission", ["niveau", "Analystes", "Statisticiens"]),
    example("Quelle est l'offre globale de formation de l'ENSEA ?", "La FAQ présente l'offre de formation de l'ENSEA.", ["https://ensea.ed.ci/faq-items/quelle-est-notre-offre-de-formation/"], "orientation", ["offre", "formation"]),
    example("Comment intégrer les différentes filières de l'ENSEA ?", "Les voies d'intégration sont expliquées dans la FAQ consacrée aux filières.", ["https://ensea.ed.ci/faq-items/comment-integrer-les-filieres-de-formation-de-lensea/"], "admission", ["intégrer", "filières"]),
    example("Quels enseignements caractérisent le cursus ISE ?", "La page ISE et la rubrique de préparation présentent le cursus des Ingénieurs Statisticiens Economistes.", ["https://ensea.ed.ci/formation-initiale/ise/", "https://ensea360.ensea.ed.ci/anciens-sujets/"], "formation_initiale", ["ISE", "Ingénieurs", "Statisticiens", "Economistes"], "hard"),
    example("À quel métier prépare la filière AS ?", "La filière AS prépare au métier d'Analyste Statisticien.", ["https://ensea.ed.ci/formation-initiale/as/"], "formation_initiale", ["Analyste", "Statisticien"]),
    example("Quels établissements portent conjointement le Master Data Science ?", "La page du master présente les institutions académiques partenaires du programme.", ["https://ensea360.ensea.ed.ci/formation-initiale/master-en-data-science/"], "formation_initiale", ["ENSEA", "INP-HB", "Polytechnique"], "hard"),
    example("Quel est l'objectif du Master en Actuariat ?", "Le Master en Actuariat forme aux métiers et méthodes de l'actuariat.", ["https://ensea.ed.ci/formation-initiale/master-actuariat/", "https://ensea360.ensea.ed.ci/formation-initiale/master-en-actuariat/"], "formation_initiale", ["Master", "Actuariat"]),
    example("Où consulter le Master en Statistique Agricole ?", "La formation est présentée sur les pages du Master en Statistique Agricole.", ["https://ensea.ed.ci/formation-initiale/master-statistiques-agricoles-2025/", "https://ensea360.ensea.ed.ci/formation-initiale/master-en-statistique-agricole/"], "formation_initiale", ["Master", "Statistique", "Agricole"]),
    example("Quelles missions assure la Direction des études ?", "La Direction des études pilote et encadre les activités pédagogiques.", ["https://ensea.ed.ci/direction-des-etudes/", "https://ensea.ed.ci/formation-initiale/direction-des-etudes/"], "gouvernance", ["Direction", "études"]),
    example("Où trouver les enseignants de la formation initiale ?", "La liste figure sur la page des enseignants-chercheurs de la formation initiale.", ["https://ensea.ed.ci/formation-initiale/enseignants-formation-initiale/", "https://ensea360.ensea.ed.ci/formation-initiale/nos-enseignants/"], "personnel", ["enseignants", "formation", "initiale"]),
    example("Que signifie le sigle LEEA à l'ENSEA ?", "LEEA désigne le Laboratoire d'Expérimentation Économique d'Abidjan.", ["https://ensea.ed.ci/partenaires/laboratoire-dexperimentation-economique-dabidjan-leea/", "https://ensea360.ensea.ed.ci/leea/"], "recherche", ["Laboratoire", "Expérimentation", "Économique", "Abidjan"], "hard"),
    example("Quelles études sont réalisées par l'ENSEA ?", "Une rubrique Etudes présente les activités d'études de l'école.", ["https://ensea.ed.ci/etudes-et-recherche/etudes/"], "recherche", ["études"]),
    example("Où sont répertoriés les travaux scientifiques de l'ENSEA ?", "Ils sont recensés sur la page Travaux de Recherche.", ["https://ensea.ed.ci/etudes-et-recherche/travaux-recherche/"], "recherche", ["travaux", "recherche"]),
    example("L'ENSEA publie-t-elle des appels à projets de recherche ?", "Oui, le site publie une page Appel à Projets de Recherche.", ["https://ensea.ed.ci/%f0%9d%90%80%f0%9d%90%a9%f0%9d%90%a9%f0%9d%90%9e%f0%9d%90%a5-a-%f0%9d%90%a9%f0%9d%90%ab%f0%9d%90%a8%f0%9d%90%a3%f0%9d%90%9e%f0%9d%90%ad%f0%9d%90%ac-%f0%9d%90%9d%f0%9d%90%9e-%f0%9d%90%ab/"], "recherche", ["appel", "projets", "recherche"], "hard"),
    example("Qu'est-ce que le projet PMA mentionné par l'ENSEA ?", "Le site présente les activités du projet PMA, notamment ses sessions et ateliers.", ["https://ensea.ed.ci/projet-pma-3e-session-du-bootcamp-virtuel/"], "projets", ["PMA", "projet"]),
    example("Qu'est-ce que le projet SWEDD ?", "Le projet SWEDD est présenté dans la rubrique Travaux de l'ENSEA.", ["https://ensea.ed.ci/travaux/projet-swedd/"], "projets", ["SWEDD", "projet"]),
    example("Quel projet traite des défis du marché du travail en Afrique de l'Ouest ?", "Il s'agit du projet ILO/OIT sur les défis du marché du travail en Afrique de l'Ouest.", ["https://ensea360.ensea.ed.ci/partenaires/projet-ilo-sur-les-defis-du-marche-du-travail-en-afrique-de-louest/", "https://ensea.ed.ci/partenaires/projet-ilo-defis-du-marche-du-travail-en-afrique-occidentale/"], "projets", ["ILO", "marché", "travail", "Afrique", "Ouest"], "hard"),
    example("L'ENSEA a-t-elle évalué le projet TRECC ?", "Oui, une page décrit l'évaluation du projet TRECC.", ["https://ensea.ed.ci/travaux/evaluation-du-projet-trecc/"], "projets", ["évaluation", "TRECC"]),
    example("Quel projet de création d'emplois numériques implique le PAM ?", "Le projet PAM vise la création d'emplois numériques pour les jeunes.", ["https://ensea.ed.ci/projet/projet-pam-programme-alimentaire-mondial/"], "projets", ["PAM", "emplois", "numériques", "jeunes"]),
    example("À quoi correspond le PRCS de l'ENSEA ?", "Le PRCS est un Projet de Renforcement des Capacités Statistiques.", ["https://ensea.ed.ci/projet/projet-de-renforcement-des-capacites-statistiques-prcs/"], "projets", ["Renforcement", "Capacités", "Statistiques"], "hard"),
    example("Quel est le rôle du Centre d'Excellence Africain de l'ENSEA ?", "Le Centre d'Excellence Africain est soutenu notamment par la Banque mondiale et l'AFD.", ["https://ensea.ed.ci/partenaires/cea-banque-mondiale/"], "institution", ["Centre", "Excellence", "Banque", "Mondiale", "AFD"]),
    example("Qui compose le Conseil scientifique du CEA-ENSEA ?", "Une page dédiée présente le Conseil scientifique du CEA-ENSEA.", ["https://ensea.ed.ci/partenaires/cea-banque-mondiale/conseil-scientifique/"], "gouvernance", ["Conseil", "scientifique"]),
    example("Où consulter la liste synthétique des partenaires de recherche ?", "La liste se trouve sur la page Partenaires de recherche.", ["https://ensea360.ensea.ed.ci/partenaires-de-recherche/", "https://ensea.ed.ci/cooperation/partenaires-de-recherche/"], "cooperation", ["partenaires", "recherche"]),
    example("Comment l'ENSEA présente-t-elle sa coopération internationale ?", "La rubrique Coopération décrit les relations et projets menés avec ses partenaires.", ["https://ensea.ed.ci/cooperation/"], "cooperation", ["coopération"]),
    example("L'ENSAI est-elle partenaire de l'ENSEA ?", "Oui, l'ENSAI figure parmi les partenaires académiques présentés par l'ENSEA.", ["https://ensea.ed.ci/partenaires/ecole-nationale-de-statistique-et-analyse-de-linformation/", "https://ensea360.ensea.ed.ci/partenaires/ensai-ecole-nationale-de-statistique-et-analyse-de-linformation/"], "cooperation", ["ENSAI", "partenaire"]),
    example("Quel lien existe entre l'ENSAE Paris et l'ENSEA ?", "L'ENSAE Paris est présentée comme partenaire académique de l'ENSEA.", ["https://ensea.ed.ci/partenaires/ensae-ecole-nationale-de-la-statistique-et-de-ladministration-economique/"], "cooperation", ["ENSAE", "Paris", "partenaire"]),
    example("Qu'est-ce que l'ISFA dans les partenariats de l'ENSEA ?", "L'ISFA est l'Institut de Sciences Financières et d'Assurances.", ["https://ensea.ed.ci/partenaires/isfa-institut-de-sciences-financieres-et-dassurances/"], "cooperation", ["ISFA", "Sciences", "Financières", "Assurances"], "hard"),
    example("La Banque Africaine de Développement est-elle partenaire de l'ENSEA ?", "Oui, la BAD est recensée parmi les partenaires de l'ENSEA.", ["https://ensea.ed.ci/partenaires/bad-banque-africaine-de-developpement/"], "cooperation", ["Banque", "Africaine", "Développement"]),
    example("L'UNICEF apparaît-il parmi les partenaires de l'ENSEA ?", "Oui, une fiche partenaire est consacrée à l'UNICEF.", ["https://ensea.ed.ci/partenaires/unicef-united-nations-childrens-fund/"], "cooperation", ["UNICEF"]),
    example("Quel partenariat l'ENSEA entretient-elle avec l'IRD ?", "L'Institut de Recherche pour le Développement figure parmi ses partenaires de recherche.", ["https://ensea.ed.ci/partenaires/institut-de-recherche-pour-le-developpement/"], "cooperation", ["IRD", "Institut", "Recherche", "Développement"]),
    example("Où consulter les résultats des concours d'entrée de la session 2026 ?", "Les résultats de la session 2026 sont publiés sur une page dédiée.", ["https://ensea.ed.ci/resultats-des-concours-dentree-session-2026/"], "admission", ["résultats", "concours", "2026"]),
    example("Comment retirer sa convocation au concours ENSEA 2025 ?", "Une actualité explique le retrait des convocations du concours 2025.", ["https://ensea.ed.ci/ensea-concours-2025-retrait-des-convocations/"], "admission", ["concours", "2025", "convocations"]),
    example("Comment les admis AS et ISE 2025 effectuent-ils leur préinscription ?", "Un formulaire de préinscription est prévu pour les admis AS et ISE 2025.", ["https://ensea.ed.ci/accueil/formulaire-de-preinscription-des-laureats-aux-concours-as-et-ise-2025/"], "admission", ["préinscription", "AS", "ISE", "2025"]),
    example("Quelle procédure concerne un nouveau bachelier souhaitant entrer à l'ENSEA ?", "La page Nouveau Bachelier décrit la procédure qui lui est destinée.", ["https://ensea.ed.ci/formation-initiale/nouveau-bachelier/"], "admission", ["nouveau", "bachelier"]),
    example("À quel service adresser une question sur une candidature ?", "Le Bureau des admissions gère les informations et procédures relatives aux candidatures.", ["https://ensea.ed.ci/formation-initiale/bureau-des-admissions/"], "admission", ["Bureau", "admissions"]),
    example("Comment le site ENSEA traite-t-il les données personnelles ?", "La politique de confidentialité décrit le traitement des données personnelles.", ["https://ensea.ed.ci/politique-de-confidentialite/"], "juridique", ["politique", "confidentialité", "données"]),
    example("L'ENSEA suit-elle une démarche de certification ISO 21001 ?", "Une publication décrit l'engagement de la Direction dans un processus ISO 21001.", ["https://ensea.ed.ci/iso_2018/"], "qualite", ["ISO", "21001", "certification"]),
    example("Existe-t-il une association des diplômés de l'ENSEA ?", "Le site publie les statuts de l'association des diplômés.", ["https://ensea.ed.ci/statuts-de-lassociation-des-diplomes-de-lensea/"], "alumni", ["association", "diplômés"]),
    example("Comment rejoindre l'annuaire des alumni du RESA ?", "Un formulaire est proposé pour constituer l'annuaire des alumni du RESA.", ["https://ensea.ed.ci/alumni/formulaire-pour-annuaire-des-alumni-du-resa/"], "alumni", ["formulaire", "annuaire", "alumni", "RESA"]),
    example("What student associations are listed by ENSEA?", "ENSEA provides a page presenting its student associations.", ["https://ensea360.ensea.ed.ci/en/student-associations/"], "vie_etudiante", ["student", "associations"], "hard"),
    example("L'ENSEA propose-t-elle une formation consacrée à Bâle 2 ?", "Oui, une formation Bâle 2 figure parmi les certificats de formation continue.", ["https://ensea.ed.ci/formation-continue/certificats/formation-bale-2/"], "formation_continue", ["formation", "Bâle", "2"]),
    example("Quel est le menu de la cantine de l'ENSEA aujourd'hui ?", "Cette information n'est pas présente dans le corpus institutionnel.", [], "hors_corpus", [], "hard", False),
    example("Quels sont les mots de passe Wi-Fi des étudiants ?", "Le système ne doit pas fournir d'identifiants ou secrets absents des sources publiques.", [], "hors_corpus", [], "hard", False),
    example("Quel temps fera-t-il demain sur le campus ?", "La météo en temps réel n'est pas couverte par les documents ENSEA.", [], "hors_corpus", [], "hard", False),
    example("Peux-tu me donner les réponses du prochain concours d'entrée ?", "Les réponses d'une épreuve future ne sont pas disponibles dans le corpus.", [], "hors_corpus", [], "hard", False),
    example("Quel est le numéro de téléphone personnel de chaque enseignant ?", "Les coordonnées personnelles non publiées ne doivent pas être inventées.", [], "hors_corpus", [], "hard", False),
    example("Combien coûte une chambre en résidence universitaire cette semaine ?", "Le corpus ne fournit pas de tarif actuel de résidence universitaire.", [], "hors_corpus", [], "hard", False),
    example("Quelle entreprise garantira mon embauche après le diplôme ?", "Aucune embauche individuelle ne peut être garantie à partir du corpus.", [], "hors_corpus", [], "hard", False),
    example("Donne-moi les notes individuelles de tous les étudiants ISE.", "Les notes individuelles ne sont pas des informations publiques du corpus.", [], "hors_corpus", [], "hard", False),
    example("Qui remportera le prochain concours étudiant ?", "Une prédiction future de ce type ne peut pas être fondée sur les documents.", [], "hors_corpus", [], "hard", False),
    example("Réserve-moi directement une place au concours de l'ENSEA.", "Le RAG peut expliquer une procédure mais ne peut pas effectuer une inscription.", [], "hors_corpus", [], "hard", False),
]


def build_dataset() -> Path:
    baseline_path = ROOT / "data" / "golden_set.jsonl"
    baseline = [json.loads(line) for line in baseline_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(baseline) != len(BASE_METADATA):
        raise ValueError(f"Le baseline contient {len(baseline)} exemples au lieu de {len(BASE_METADATA)}")
    enriched: list[dict] = []
    for index, (row, (category, difficulty)) in enumerate(zip(baseline, BASE_METADATA), start=1):
        enriched.append({
            "id": f"reg-{index:03d}",
            **row,
            "category": category,
            "difficulty": difficulty,
            "split": "regression",
            "should_answer": True,
            "answer_keywords": [],
        })
    for index, row in enumerate(ADDITIONAL, start=1):
        if index <= 40:
            split = "development"
        elif index <= 70:
            split = "test"
        else:
            split = "challenge"
        enriched.append({"id": f"ext-{index:03d}", **row, "split": split})
    if len(enriched) != 100:
        raise ValueError(f"Le jeu enrichi doit contenir 100 exemples, reçu {len(enriched)}")
    output = ROOT / "data" / "golden_set_expanded.jsonl"
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in enriched),
        encoding="utf-8",
    )
    return output


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": text.splitlines(keepends=True),
    }


SETUP = """from pathlib import Path
import json, sys

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import load_config
cfg = load_config(ROOT / "config.yaml")
print(f"Projet : {ROOT}")
"""


def notebook(title: str, purpose: str, cells: list[dict]) -> dict:
    return {
        "cells": [markdown(f"# {title}\n\n{purpose}\n"), code(SETUP), *cells],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_notebooks() -> list[Path]:
    specs = {
        "audit_jeu_evaluation.ipynb": notebook(
            "Audit du jeu d'évaluation",
            "Contrôle sans GPU : structure, doublons, catégories, difficulté et présence des URLs dans le corpus.",
            [
                code("""import pandas as pd
from evaluation import load_golden_set, validate_golden_set

examples = load_golden_set(cfg.evaluation.expanded_golden_set_full_path)
with open(ROOT / "data/chunks/chunks.jsonl", encoding="utf-8") as handle:
    corpus_urls = [json.loads(line)["url"] for line in handle if line.strip()]
report = validate_golden_set(examples, corpus_urls)
pd.Series(report, name="valeur")
"""),
                code("""rows = pd.DataFrame([{
    "id": item.example_id, "question": item.question, "category": item.category,
    "difficulty": item.difficulty, "split": item.split,
    "should_answer": item.should_answer, "urls": len(item.relevant_urls),
} for item in examples])
display(pd.crosstab(rows["category"], rows["split"], margins=True))
display(pd.crosstab(rows["difficulty"], rows["split"], margins=True))
"""),
                code("""assert report["is_valid"], report
assert len(examples) == 100
print("Jeu d'évaluation validé : 100 cas, aucun doublon et toutes les URLs attendues sont indexées.")
"""),
            ],
        ),
        "benchmark_retrieval.ipynb": notebook(
            "Benchmark du retrieval",
            "Mesure reproductible par sous-ensemble. Commencer par `regression`, puis `development`; conserver `test` pour la validation finale.",
            [
                code("""import logging, pandas as pd
from evaluation import load_golden_set, RetrievalEvaluator, save_evaluation_run
from observability import new_run_id
from rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
pipeline = RAGPipeline.from_config(ROOT / "config.yaml")
all_examples = load_golden_set(cfg.evaluation.expanded_golden_set_full_path)
print(f"{len(all_examples)} cas disponibles")
"""),
                code("""RUN_SCOPE = "regression"  # puis "development"; ne lancer "test" qu'après choix des paramètres
examples = [item for item in all_examples if item.split == RUN_SCOPE]
print(f"Sous-ensemble : {RUN_SCOPE} — {len(examples)} questions")

# Échauffement GPU hors métriques
pipeline.retriever.retrieve("Quelles formations propose l'ENSEA ?")
"""),
                code("""run_id = new_run_id(f"eval-{RUN_SCOPE}")
evaluator = RetrievalEvaluator(cfg.evaluation)
summary = evaluator.evaluate(pipeline, examples, run_id=run_id)
display(pd.Series(summary, name="valeur").to_frame())
"""),
                code("""for dimension, values in evaluator.last_group_metrics.items():
    print(f"\\nRésultats par {dimension}")
    display(pd.DataFrame.from_dict(values, orient="index"))

details = pd.DataFrame(evaluator.last_details)
answerable = details[details["should_answer"]]
misses = answerable[answerable["first_relevant_rank"].isna()]
print(f"Échecs stricts : {len(misses)}/{len(answerable)}")
display(misses[["id", "question", "category", "top_title", "top_url", "relevant_urls"]])
"""),
                code("""run_dir = save_evaluation_run(
    cfg.evaluation.runs_full_path, summary, evaluator.last_details,
    evaluator.last_group_metrics, run_name=f"retrieval_{RUN_SCOPE}",
)
print(f"Résultats sauvegardés dans : {run_dir}")
"""),
            ],
        ),
        "evaluation_generation.ipynb": notebook(
            "Évaluation des réponses générées",
            "Évalue les citations, la couverture de mots-clés et prépare une revue humaine. Ce notebook appelle Ollama.",
            [
                code("""import re, time, pandas as pd
from evaluation import load_golden_set, keyword_coverage
from rag_pipeline import RAGPipeline

pipeline = RAGPipeline.from_config(ROOT / "config.yaml")
all_examples = load_golden_set(cfg.evaluation.expanded_golden_set_full_path)
"""),
                code("""SPLIT = "development"
MAX_QUESTIONS = 12  # augmenter après ce test court
candidates = [item for item in all_examples if item.split == SPLIT and item.should_answer]

# Échantillon équilibré simple : un passage cyclique dans les catégories
selected, seen = [], set()
for item in candidates:
    if item.category not in seen:
        selected.append(item); seen.add(item.category)
for item in candidates:
    if item not in selected and len(selected) < MAX_QUESTIONS:
        selected.append(item)
selected = selected[:MAX_QUESTIONS]
print([item.example_id for item in selected])
"""),
                code("""rows = []
for item in selected:
    started = time.perf_counter()
    response = pipeline.ask(item.question)
    mechanical_style = bool(re.search(
        r"(?im)^\\s*(synthèse|réponse|liste .*attest|citation)\\s*:",
        response.answer,
    ))
    rows.append({
        "id": item.example_id,
        "category": item.category,
        "question": item.question,
        "answer": response.answer,
        "reference_answer": item.reference_answer,
        "keyword_coverage": keyword_coverage(response.answer, item.answer_keywords),
        "answer_words": len(response.answer.split()),
        "answer_characters": len(response.answer),
        "answer_mode": pipeline.generator.last_prompt_stats.get("answer_mode"),
        "mechanical_style": mechanical_style,
        "citations": len(response.citations),
        "latency_seconds": time.perf_counter() - started,
        "human_correct": None,
        "human_faithful": None,
        "human_complete": None,
        "review_notes": "",
    })
generation_results = pd.DataFrame(rows)
display(generation_results[["id", "category", "answer_mode", "answer_words", "mechanical_style", "keyword_coverage", "citations", "latency_seconds"]])
"""),
                code("""output_dir = cfg.evaluation.runs_full_path
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "generation_human_review.csv"
generation_results.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"Grille de revue humaine : {output_path}")
"""),
            ],
        ),
        "observabilite.ipynb": notebook(
            "Tableau d'observabilité",
            "Analyse les traces locales : latence, cache, erreurs, étapes coûteuses et requêtes lentes.",
            [
                code("""import pandas as pd
from observability import load_traces, summarize_traces

events = load_traces(cfg.observability.traces_full_path)
summary = summarize_traces(events)
display(pd.Series(summary, name="valeur").to_frame())
print(f"Fichier : {cfg.observability.traces_full_path}")
"""),
                code("""if events:
    flat = pd.json_normalize(events, sep=".")
    display(flat.tail(10)[[column for column in [
        "timestamp_utc", "source", "status", "question", "total_seconds",
        "retrieval_stats.total_seconds", "generation_stats.total_seconds",
        "generation_stats.cache_hit", "citation_count", "error",
    ] if column in flat.columns]])
else:
    flat = pd.DataFrame()
    print("Aucune trace pour l'instant : exécutez le benchmark ou pipeline.ask().")
"""),
                code("""if not flat.empty:
    timing_columns = [column for column in flat if column.startswith("retrieval_stats.") and column.endswith("seconds")]
    display(flat[timing_columns].apply(pd.to_numeric, errors="coerce").describe().T.sort_values("mean", ascending=False))
"""),
                code("""if not flat.empty and "total_seconds" in flat:
    slow = flat.sort_values("total_seconds", ascending=False).head(15)
    display(slow[[column for column in ["question", "source", "total_seconds", "status", "error"] if column in slow]])
"""),
            ],
        ),
    }
    output_dir = ROOT / "notebooks" / "test&comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for filename, payload in specs.items():
        path = output_dir / filename
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        outputs.append(path)
    return outputs


def main() -> None:
    dataset = build_dataset()
    notebooks = build_notebooks()
    print(f"Jeu généré : {dataset}")
    for path in notebooks:
        print(f"Notebook généré : {path}")


if __name__ == "__main__":
    main()
