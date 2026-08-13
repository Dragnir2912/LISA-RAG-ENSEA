# Parcours RAG, étape par étape

Commencez par `00_PARCOURS_RAG.ipynb`, puis exécutez les notebooks de `01` à
`13` dans l’ordre. Utilisez le noyau Python `.venv`, arrêtez l’application
Docker et fermez les autres kernels avant les notebooks 05 à 09 afin d’éviter
les conflits d’index et de mémoire.

Chaque notebook répond à quatre questions :

1. Quelles données entrent dans cette étape ?
2. Quel composant du projet intervient ?
3. Quels contrôles permettent de valider le résultat ?
4. Quel artefact est transmis à l'étape suivante ?

Les notebooks chargent les modules de `src/` : ils montrent donc le fonctionnement réel du projet sans recopier une seconde implémentation simplifiée du RAG.

Les notebooks 03 à 07 constituent la reconstruction officielle : ils produisent
successivement les records PDF et Excel locaux, les faits structurés des
tableaux, le corpus nettoyé, les chunks narratifs et tabulaires, les matrices de
vecteurs et les index actifs. Exécutez-les dans l’ordre et attendez la fin de
chaque notebook avant de poursuivre. Les dossiers de sortie sont créés
automatiquement, mais `data/sources/` doit déjà contenir les documents originaux.

Le notebook 08 valide ensuite le retrieval sur des faits de contrôle, dont ceux
du document *Chiffres clés CIV 2025*. Ces contrôles ne sont pas des règles de
production propres à deux questions : ils vérifient que les lignes de tableaux,
leurs périodes, unités et pages ont bien traversé toute la chaîne. Le notebook
10 mesure ensuite le retrieval et la génération sur les golden sets.

Les drapeaux de collecte réseau sont désactivés par défaut. Les drapeaux de
reconstruction locale 03 à 07 sont activés dans la version actuelle : leur
exécution remplace les artefacts actifs. En cas d’interruption pendant les
embeddings ou l’indexation, relancez le notebook concerné en entier.

Le sommaire détaillé se trouve dans [`00_PARCOURS_RAG.ipynb`](00_PARCOURS_RAG.ipynb) et dans [`../README.md`](../README.md).
