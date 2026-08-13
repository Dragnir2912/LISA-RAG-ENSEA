# Lancer LISA localement avec Docker sous Windows

Ce guide est la procédure de référence pour utiliser l’application sur un PC
Windows. Docker exécute des conteneurs Linux. Le profil **CPU** fonctionne sans
carte graphique ; le profil **GPU** est facultatif et nécessite une carte NVIDIA
correctement exposée à Docker Desktop.

Les commandes doivent être exécutées dans **PowerShell**, depuis la racine du
projet, c’est-à-dire le dossier qui contient `compose.test.yaml`.

## 1. Prérequis à installer une seule fois

- Windows 10 ou 11 avec WSL 2 activé ;
- Docker Desktop configuré avec le moteur WSL 2 et les conteneurs Linux ;
- au moins 20 Go d’espace libre pour les images et modèles ;
- le dossier complet du projet, notamment `data/` et `config.yaml`.

Après le démarrage de Docker Desktop, vérifier :

```powershell
docker version
docker compose version
```

Les deux commandes doivent afficher une version sans erreur de connexion.

## 2. Préparation initiale avec Internet

Cette étape construit les images, crée les secrets locaux et télécharge les
modèles. Elle demande une connexion Internet **une seule fois par machine**.

Si Windows bloque les scripts PowerShell, autoriser uniquement la session
courante :

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

Préparer le profil CPU :

```powershell
.\deployment\local\prepare.ps1 -Profile cpu
```

Ou préparer le futur lancement NVIDIA :

```powershell
.\deployment\local\prepare.ps1 -Profile gpu
```

Le script :

1. crée `.env.local` depuis `.env.local.example` s’il n’existe pas ;
2. crée deux secrets aléatoires dans `secrets/` ;
3. construit les images de l’API et du frontend ;
4. télécharge PostgreSQL, Ollama, BGE-M3, le reranker et `qwen3.5:9b` ;
5. conserve les caches dans `models/` pour les démarrages hors ligne.

Il ne supprime ni `data/`, ni les embeddings, ni les index existants.

## 3. Démarrer l’application

### Profil CPU, compatible avec toutes les machines

```powershell
.\deployment\local\start.ps1 -Profile cpu
```

### Profil GPU NVIDIA

```powershell
.\deployment\local\start.ps1 -Profile gpu
```

Le fichier `compose.gpu.yaml` applique automatiquement les réglages GPU : il
n’est pas nécessaire de modifier `.env.local` en changeant de profil.

Ouvrir ensuite [http://localhost:8080](http://localhost:8080). Le premier
chargement de Qwen peut durer plusieurs minutes sur CPU ; les appels suivants
sont plus rapides tant que le modèle reste chargé.

Le démarrage courant utilise `--no-build`. Pour reconstruire les images après
une modification du code, utiliser :

```powershell
.\deployment\local\start.ps1 -Profile cpu -Build
```

La reconstruction peut nécessiter Internet si le cache Docker est incomplet.

## 4. Voir l’état ou les journaux

État des services :

```powershell
.\deployment\local\status.ps1
```

Journaux en continu :

```powershell
.\deployment\local\status.ps1 -Follow
```

Quitter les journaux avec `Ctrl+C` ne coupe pas l’application.

## 5. Arrêter sans perdre les données

Après un lancement CPU :

```powershell
.\deployment\local\stop.ps1 -Profile cpu
```

Après un lancement GPU :

```powershell
.\deployment\local\stop.ps1 -Profile gpu
```

Les comptes, conversations, modèles, documents et index sont conservés. Ne pas
utiliser `docker compose down --volumes` : cette option supprimerait la base
PostgreSQL locale.

## 6. Fonctionnement hors ligne

Après une préparation initiale réussie :

1. laisser `RAG_OFFLINE_MODE=1` dans `.env.local` ;
2. couper Internet ;
3. lancer avec `start.ps1` sans l’option `-Build` ;
4. ouvrir une conversation et interroger les documents ;
5. tester l’ajout puis l’indexation d’un PDF local.

Le chat, les comptes, les PDF, les embeddings et les index restent locaux. Les
collectes Web, Facebook et tout service d’envoi d’e-mail extérieur sont
indisponibles hors ligne par nature ; l’interface les signale au lieu de lancer
une tâche vouée à échouer.

## 7. Dépannage

- **Docker est introuvable** : démarrer ou installer Docker Desktop.
- **Docker Desktop ne répond pas** : attendre « Engine running », puis relancer.
- **Un modèle est absent** : reconnecter temporairement la machine et relancer
  `prepare.ps1` ; les fichiers déjà présents sont réutilisés.
- **Le port 8080 est occupé** : modifier `LISA_HTTP_PORT` dans `.env.local`.
- **Le GPU n’est pas reconnu** : revenir à `-Profile cpu`, puis vérifier les
  pilotes NVIDIA et l’intégration WSL de Docker Desktop.
- **La réponse CPU est lente** : c’est attendu pour un modèle local de 9 milliards
  de paramètres ; aucune donnée n’est envoyée vers un service externe.
- **Un service redémarre en boucle** : consulter
  `.\deployment\local\status.ps1 -Follow` et corriger la première erreur affichée.

Pour une installation Linux multi-utilisateur avec HTTPS et sauvegardes, suivre
le [guide du serveur de test](deploiement_test.md).
