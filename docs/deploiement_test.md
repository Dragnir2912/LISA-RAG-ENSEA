# Déployer LISA sur un serveur de test Linux

Ce guide cible **Ubuntu Server 22.04 ou 24.04 LTS en x86-64**. Toutes les
commandes sont des commandes Linux. La pile fonctionne sans GPU et, une fois
préparée, sans connexion Internet.

Toutes les commandes qui utilisent `deployment/linux/*.sh` doivent être lancées
depuis la racine `/opt/lisa`. Les scripts prennent deux paramètres dans cet
ordre : le profil de calcul (`cpu` ou `gpu`), puis le mode HTTPS (`offline`,
`public` ou `none`). Pour un serveur de recette ENSEA, le choix sûr par défaut
est `cpu offline`.

## 1. Prérequis à installer avant de copier l’application

### Configuration minimale

| Élément | Minimum CPU | Recommandé avec GPU |
|---|---:|---:|
| CPU | 8 cœurs x86-64 | 8 cœurs x86-64 |
| RAM | 32 Gio | 32 Gio |
| Disque libre | 150 Gio | 150 Gio |
| GPU | aucun | NVIDIA, 8 Gio de VRAM ou plus |
| Système | Ubuntu Server 22.04/24.04 LTS | idem |

Le profil CPU est pleinement fonctionnel ; l’embedding, le reranking et la
génération seront simplement plus lents. **NVIDIA n’est donc jamais un
prérequis obligatoire.**

### Paquets système obligatoires

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg openssl tar
```

### Docker Engine, Buildx et Compose

```bash
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

. /etc/os-release
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${UBUNTU_CODENAME:-$VERSION_CODENAME} stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
```

Fermer la session SSH, se reconnecter, puis vérifier :

```bash
docker version
docker compose version
docker run --rm hello-world
```

### NVIDIA, uniquement si le serveur possède un GPU

Ne pas exécuter cette section sur un serveur CPU.

```bash
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers install
sudo reboot
```

Après reconnexion, `nvidia-smi` doit fonctionner. Installer ensuite le NVIDIA
Container Toolkit :

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

La version exacte du pilote doit être compatible avec le matériel et la
distribution du serveur. Le script de contrôle LISA ne considère pas son
absence comme une erreur :

```bash
chmod +x deployment/linux/*.sh
deployment/linux/check_prerequisites.sh
```

## 2. Ce qui fonctionne réellement hors ligne

Une fois les images et modèles préparés, les fonctions suivantes ne font aucun
appel Internet :

- comptes, invités, sessions et conversations ;
- questions/réponses avec Qwen via Ollama local ;
- BGE-M3, BGE Reranker, LanceDB et BM25 ;
- import de PDF locaux, nettoyage, chunks, embeddings et indexation ;
- administration, métriques, sauvegardes et historique.

Les collectes de sites Web, Facebook/Apify et l’envoi SMTP sont, par nature,
des fonctions réseau. `RAG_OFFLINE_MODE=1` les bloque avec un message clair au
lieu de laisser un traitement échouer. La récupération de mot de passe reste
possible par un administrateur ; l’envoi automatique d’un e-mail exige un
relais SMTP joignable.

## 3. Structure Docker retenue

```text
compose.test.yaml           base CPU, PostgreSQL, Ollama, API, worker, frontend
compose.gpu.yaml            extension NVIDIA facultative
compose.tls.yaml            HTTPS public avec certificats automatiques
compose.tls.offline.yaml    HTTPS avec certificats locaux déjà disponibles
deployment/linux/           contrôle, préparation et paquet hors ligne
data/                       sources, chunks, embeddings, LanceDB et BM25
models/huggingface/         BGE-M3 et BGE Reranker
models/ollama/              Qwen 3.5 9B
secrets/                    mots de passe, clé applicative, certificats
```

PostgreSQL, l’API, le worker et Ollama ne publient aucun port. Le frontend est
lié à `127.0.0.1:8080`; seul Caddy publie 80/443 quand une extension TLS est
activée.

## 4. Préparer le répertoire serveur

```bash
sudo mkdir -p /opt/lisa
sudo chown "$USER":"$USER" /opt/lisa
cd /opt/lisa
```

Copier ensuite le projet dans `/opt/lisa`, y compris `data/`. Les embeddings et
index déjà validés doivent rester présents.

```bash
cd /opt/lisa
test -f compose.test.yaml
test -f config.yaml
test -d data/embeddings
test -d data/economic
test -d data/facebook
chmod +x deployment/linux/*.sh
deployment/linux/check_prerequisites.sh
```

## 5. Configurer l’environnement et les secrets

```bash
cd /opt/lisa
cp .env.test.example .env.test
mkdir -p secrets
chmod 700 secrets
umask 077
openssl rand -base64 48 | tr -d '\n' > secrets/postgres_password.txt
openssl rand 32 | openssl base64 -A | tr '+/' '-_' > secrets/application_key.txt
chmod 600 .env.test secrets/postgres_password.txt secrets/application_key.txt
```

Éditer `.env.test` et remplacer au minimum le domaine et les trois URL :

```bash
nano .env.test
```

Pour un serveur CPU et hors ligne, conserver :

```dotenv
RAG_FORCE_CPU=1
RAG_OLLAMA_NUM_GPU=0
RAG_OFFLINE_MODE=1
```

Pour un serveur GPU, utiliser plutôt :

```dotenv
RAG_FORCE_CPU=0
RAG_OLLAMA_NUM_GPU=999
RAG_OFFLINE_MODE=1
```

Ne jamais placer les secrets applicatifs dans `.env.test` et ne jamais les
versionner.

## 6. Première préparation avec une connexion temporaire

Cette phase est la seule qui télécharge les images Docker, les paquets Python et
les trois modèles. Une fois terminée, la connexion peut être coupée.

```bash
cd /opt/lisa
export LISA_IMAGE_TAG=test
deployment/linux/prepare_runtime_assets.sh .env.test
deployment/linux/verify_offline_ready.sh .env.test
```

Les modèles sont conservés dans `models/`, pas dans un cache personnel. Le
backend démarre ensuite avec `HF_HUB_OFFLINE=1` et
`TRANSFORMERS_OFFLINE=1`. Ollama démarre avec `OLLAMA_NO_CLOUD=1`.

## 7. Installer sur un serveur qui n’a jamais accès à Internet

Sur une autre machine Linux connectée contenant le projet et `data/`, créer le
paquet :

```bash
cd /opt/lisa
cp .env.test.example .env.test
chmod +x deployment/linux/*.sh
export LISA_IMAGE_TAG=test
deployment/linux/prepare_offline_bundle.sh .env.test dist/lisa-offline
```

Le dossier `dist/lisa-offline` contient les images Docker, l’application,
**tous les index et embeddings**, les modèles, et leurs empreintes SHA-256. Il
ne contient aucun secret.

Copier ce dossier sur le serveur déconnecté, puis :

```bash
sudo mkdir -p /opt/lisa
sudo chown "$USER":"$USER" /opt/lisa
cd /opt/lisa
(cd /mnt/transfert/lisa-offline && sha256sum -c SHA256SUMS)
tar -xzf /mnt/transfert/lisa-offline/application.tar.gz
tar -xzf /mnt/transfert/lisa-offline/data.tar.gz
tar -xzf /mnt/transfert/lisa-offline/models.tar.gz
docker load -i /mnt/transfert/lisa-offline/docker-images.tar
chmod +x deployment/linux/*.sh
```

Créer ensuite `.env.test` et les secrets avec la section 5, puis valider :

```bash
export LISA_IMAGE_TAG=test
deployment/linux/verify_offline_ready.sh .env.test
```

Cette validation utilise uniquement les fichiers locaux et échoue avant le
démarrage si un modèle ou un index manque.

## 8. Choisir le HTTPS

### Réseau connecté avec un domaine public

`compose.tls.yaml` demande et renouvelle automatiquement le certificat :

```bash
docker compose --env-file .env.test \
  -f compose.test.yaml -f compose.tls.yaml config >/dev/null
```

### Réseau isolé ou certificat fourni par l’ENSEA

```bash
mkdir -p secrets/tls
cp /chemin/vers/fullchain.pem secrets/tls/fullchain.pem
cp /chemin/vers/privkey.pem secrets/tls/privkey.pem
chmod 600 secrets/tls/*.pem
docker compose --env-file .env.test \
  -f compose.test.yaml -f compose.tls.offline.yaml config >/dev/null
```

Pour une recette interne uniquement, un certificat autosigné peut être créé :

```bash
mkdir -p secrets/tls
openssl req -x509 -newkey rsa:4096 -sha256 -nodes -days 825 \
  -keyout secrets/tls/privkey.pem \
  -out secrets/tls/fullchain.pem \
  -subj "/CN=lisa-test.ensea.local" \
  -addext "subjectAltName=DNS:lisa-test.ensea.local"
chmod 600 secrets/tls/*.pem
```

Le certificat ou son autorité doit être ajouté au magasin de confiance des
postes de test pour supprimer l’avertissement du navigateur.

## 9. Lancer l’application

### Serveur CPU

```bash
cd /opt/lisa
deployment/linux/start.sh cpu offline
```

### Serveur NVIDIA

```bash
cd /opt/lisa
deployment/linux/start.sh gpu offline
```

Avec un domaine public joignable et Caddy autorisé à obtenir son certificat :

```bash
deployment/linux/start.sh cpu public
```

Le mode `none` ne publie que le frontend sur `127.0.0.1`. Il sert au diagnostic
local ou derrière un reverse proxy déjà administré ; il ne doit pas exposer
directement LISA sur Internet.

Le script refuse de démarrer si l’environnement, les secrets, les modèles ou
l’index institutionnel sont absents. Il utilise `--no-build`, donc aucun
téléchargement implicite n’est tenté pendant un démarrage hors ligne.

## 10. Vérifier le démarrage

```bash
deployment/linux/status.sh
LISA_DOMAIN=$(sed -n 's/^LISA_DOMAIN=//p' .env.test)
curl -k --resolve "${LISA_DOMAIN}:443:127.0.0.1" \
  "https://${LISA_DOMAIN}/api/health"
```

Vérifier ensuite dans l’interface : inscription, invité, connexion,
conversation, citations, import d’un PDF local et panneau administrateur. Dans
le profil CPU, la page Outils doit indiquer « CPU uniquement ».

Pour confirmer qu’aucun téléchargement implicite n’a lieu, couper temporairement
la route Internet du serveur puis poser une question documentaire. Les logs ne
doivent contenir aucune tentative Hugging Face ou Ollama Cloud.

## 11. Arrêter et redémarrer

Arrêt sans supprimer les comptes, index, modèles ni la base PostgreSQL :

```bash
deployment/linux/stop.sh cpu offline
```

Redémarrage CPU :

```bash
deployment/linux/start.sh cpu offline
```

Pour un profil GPU ou un certificat public, réutiliser exactement les mêmes
paramètres qu’au démarrage (`gpu offline`, `cpu public`, etc.).

Ne jamais ajouter `--volumes` à la commande d’arrêt courante : cela supprimerait
le volume PostgreSQL.

## 12. Sauvegarder

```bash
cd /opt/lisa
mkdir -p backups
docker compose --env-file .env.test -f compose.test.yaml exec -T postgres \
  pg_dump -U lisa -d lisa_rag -Fc \
  > "backups/lisa_$(date +%F_%H%M).dump"
tar -czf "backups/lisa_data_$(date +%F_%H%M).tar.gz" data
tar -czf "backups/lisa_models_$(date +%F_%H%M).tar.gz" models
```

Conserver les sauvegardes sur un support distinct. Les secrets doivent être
sauvegardés séparément dans un coffre-fort.

## 13. Diagnostic rapide

```bash
docker compose --env-file .env.test -f compose.test.yaml ps
docker compose --env-file .env.test -f compose.test.yaml logs --since=30m api worker ollama postgres
docker stats --no-stream
df -h
free -h
```

Sur GPU seulement :

```bash
nvidia-smi
docker compose --env-file .env.test -f compose.test.yaml exec ollama ollama ps
```

Causes fréquentes :

- **modèle absent** : refaire la préparation connectée ou retransférer
  `models.tar.gz` ;
- **cache BGE absent** : `verify_offline_ready.sh` le signale avant démarrage ;
- **réponse très lente sur CPU** : comportement normal pour Qwen 9B ;
- **collecte refusée** : comportement normal avec `RAG_OFFLINE_MODE=1` ;
- **HTTPS en erreur hors ligne** : utiliser `compose.tls.offline.yaml`, pas le
  certificat automatique ;
- **permission refusée sur `data/` ou `models/`** : rendre ces dossiers lisibles
  par Docker sans les rendre publics.

## 14. Checklist avant recette

- [ ] Docker Engine et Docker Compose validés.
- [ ] Au moins 32 Gio de RAM et 150 Gio de disque disponibles.
- [ ] `data/` contient chunks, embeddings, LanceDB et BM25.
- [ ] `models/huggingface/` et `models/ollama/` sont présents.
- [ ] `verify_offline_ready.sh` réussit.
- [ ] `.env.test` force CPU ou active explicitement NVIDIA.
- [ ] `RAG_OFFLINE_MODE=1` est actif si le serveur n’a pas Internet.
- [ ] PostgreSQL, API, worker et Ollama ne sont pas exposés publiquement.
- [ ] HTTPS et confiance du certificat sont validés.
- [ ] Une sauvegarde et une restauration ont été testées.

## Références officielles

- [Installer Docker Engine sur Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
- [Installer NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Exécuter Ollama en mode local uniquement](https://docs.ollama.com/faq)
