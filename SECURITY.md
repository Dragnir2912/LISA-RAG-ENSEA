# Politique de sécurité

## Signaler un problème

Une vulnérabilité, un secret exposé ou un contournement d’autorisation doit être
signalé **en privé** au propriétaire du dépôt ou au responsable technique ENSEA.
Ne pas créer d’issue publique contenant une procédure d’exploitation, un jeton,
des données personnelles ou un extrait de base.

Le signalement doit préciser :

- la version ou le commit concerné ;
- le composant et le rôle utilisateur touchés ;
- les conditions minimales de reproduction ;
- l’impact possible ;
- les journaux utiles, après suppression des données sensibles.

## Mesures immédiates en cas de fuite

1. révoquer ou remplacer le secret auprès du fournisseur concerné ;
2. retirer le service exposé si une exploitation active est possible ;
3. préserver les journaux nécessaires à l’analyse ;
4. corriger le code et les configurations ;
5. purger le secret de l’historique Git avant toute nouvelle publication.

Supprimer simplement un secret dans un nouveau commit ne suffit pas : les
commits précédents le conservent. Considérez toujours un secret commité comme
compromis.

## Périmètre de référence

Le profil serveur doit conserver HTTPS, les cookies sécurisés, CSRF, les hôtes
et origines autorisés, les contrôles de rôle côté API, PostgreSQL et Ollama sur
le réseau Docker privé, ainsi que les secrets montés depuis `secrets/`. Les
instructions de déploiement sûres se trouvent dans
[`docs/deploiement_test.md`](docs/deploiement_test.md).
