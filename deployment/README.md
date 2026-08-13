# Fichiers de déploiement

Ce dossier contient les scripts appelés par les guides officiels. Il ne faut pas
exécuter un script au hasard : choisir d’abord la plateforme ci-dessous.

## Windows avec Docker Desktop

Suivre [`../docs/deploiement_local.md`](../docs/deploiement_local.md).

| Script | Effet |
|---|---|
| `local/prepare.ps1` | préparation connectée initiale : secrets, images et modèles |
| `local/start.ps1` | démarrage CPU ou GPU, sans reconstruction par défaut |
| `local/status.ps1` | état puis journaux ; `-Follow` les suit en continu |
| `local/stop.ps1` | arrêt sans supprimer les volumes ni les données |

Exemple courant :

```powershell
.\deployment\local\prepare.ps1 -Profile cpu
.\deployment\local\start.ps1 -Profile cpu
.\deployment\local\stop.ps1 -Profile cpu
```

## Serveur Linux

Suivre [`../docs/deploiement_test.md`](../docs/deploiement_test.md).

| Script | Effet |
|---|---|
| `linux/check_prerequisites.sh` | contrôle Docker et, s’il existe, NVIDIA |
| `linux/prepare_runtime_assets.sh` | construit/télécharge les images et modèles |
| `linux/verify_offline_ready.sh` | vérifie qu’un lancement sans Internet est possible |
| `linux/prepare_offline_bundle.sh` | crée un paquet transférable vers un serveur isolé |
| `linux/start.sh` | démarre avec profil `cpu`/`gpu` et HTTPS `offline`/`public`/`none` |
| `linux/status.sh` | affiche l’état et les journaux ; `--follow` les suit |
| `linux/stop.sh` | arrête la même combinaison sans supprimer les volumes |

Exemple de recette CPU avec certificat local :

```bash
chmod +x deployment/linux/*.sh
deployment/linux/start.sh cpu offline
deployment/linux/status.sh --follow
deployment/linux/stop.sh cpu offline
```

## Fichiers d’infrastructure

- `backend-entrypoint.sh` attend PostgreSQL puis prépare le backend ;
- `nginx.conf` sert le frontend et transmet `/api` à FastAPI ;
- `Caddyfile` gère un certificat public ;
- `Caddyfile.offline` utilise les certificats montés depuis `secrets/tls/` ;
- `prefetch_models.py` télécharge ou contrôle le cache BGE ;
- les fichiers Compose se trouvent à la racine du projet.

Les scripts ne doivent jamais recevoir les secrets comme arguments. Les secrets
réels restent dans `secrets/`, qui est exclu de Git.
