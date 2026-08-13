# Contribuer à LISA

Ce dépôt contient le code de l’application et les exemples de configuration,
mais pas les corpus privés, modèles, index, bases, secrets ni rapports générés.
Toute contribution doit préserver cette séparation.

## Préparer l’environnement

- Pour l’application complète : suivre le [guide Docker local](docs/deploiement_local.md).
- Pour les notebooks et tests Python : suivre la section d’installation de
  [`DEPENDENCIES.md`](DEPENDENCIES.md).
- Pour comprendre les composants avant de les modifier : lire
  [`docs/architecture.md`](docs/architecture.md).

## Créer une branche

Partir d’une branche `main` à jour et utiliser un nom explicite :

```bash
git switch main
git pull --ff-only
git switch -c feature/description-courte
```

Préférer `fix/…` pour une correction et `docs/…` pour une modification de
documentation. Ne pas travailler directement sur `main` une fois le dépôt
partagé.

## Règles de modification

1. Ne jamais intégrer de mot de passe, jeton Apify, clé, certificat privé ou
   fichier `.env` réel.
2. Ne jamais ajouter `data/`, `models/`, `secrets/`, `reports/`, `dist/` ou une
   sauvegarde PostgreSQL en contournant `.gitignore`.
3. Préserver les métadonnées de source, page, période, unité et provenance des
   chunks.
4. Évaluer une modification du retrieval sur un golden set, pas uniquement sur
   la question qui a révélé le problème.
5. Ajouter ou adapter un test automatisé pour toute règle métier générale.
6. Mettre à jour le README ou le guide concerné lorsqu’une commande, variable
   d’environnement ou étape de déploiement change.

## Contrôles avant un commit

Depuis la racine, avec l’environnement `.venv` actif :

```powershell
python -m pytest -q
```

Si Docker est disponible, vérifier aussi la configuration CPU :

```powershell
docker compose --env-file .env.local.example -f compose.test.yaml config --quiet
```

Sous Linux, valider les prérequis et la préparation hors ligne :

```bash
deployment/linux/check_prerequisites.sh
deployment/linux/verify_offline_ready.sh .env.test
```

Le second contrôle suppose que `data/`, `models/` et les secrets de test sont
déjà installés sur la machine.

Enfin, examiner exactement ce qui sera versionné :

```bash
git status --short
git diff --check
git diff --cached --stat
```

La liste ne doit contenir ni document source réel, ni embedding, ni modèle, ni
base, ni secret. Ne jamais utiliser `git add -f` pour contourner une exclusion
sans revue explicite.

## Commits et revue

Écrire des commits courts et cohérents, par exemple :

```text
fix: fiabiliser le routage des questions sociales
docs: clarifier le déploiement Linux hors ligne
test: couvrir les périodes des faits tabulaires
```

Une demande de fusion doit expliquer le problème, la solution générale, les
tests effectués et l’effet observé sur la précision et la latence lorsque le RAG
est modifié.

## Vulnérabilités

Ne pas publier une vulnérabilité exploitable ni un secret dans une issue. Suivre
la procédure privée décrite dans [`SECURITY.md`](SECURITY.md).
