# Guide de Déploiement et Surveillance "On-Site" avec PyRIT

Ce document présente une sélection d'outils open-source pouvant être utilisés conjointement avec PyRIT pour des déploiements "on-site" (locaux), garantissant la sécurité, la résilience et la surveillance de vos systèmes d'IA.

## 1. Sauvegarde et Restauration avec Kopia

**Kopia** est un outil de sauvegarde open-source rapide et sécurisé qui vous permet de créer des snapshots chiffrés et dédupliqués de vos données.

- **Dépôt GitHub** : [kopia/kopia](https://github.com/kopia/kopia)

Dans un contexte PyRIT, Kopia peut être utilisé pour :
- **Sauvegarder la mémoire de PyRIT** : Protégez l'historique de vos tests et les résultats stockés (par exemple, dans une base de données SQLite ou Azure SQL) en effectuant des sauvegardes régulières.
- **Reproductibilité** : Restaurez un état spécifique de la mémoire pour reproduire des tests ou analyser des résultats passés.
- **Sécurité** : Le chiffrement de Kopia garantit que les données sensibles issues de vos tests de Red Teaming restent confidentielles.

## 2. Orchestration de Workflows avec Kestra

**Kestra** est une plateforme d'orchestration de données et de workflows open-source, déclarative et agnostique au langage.

- **Dépôt GitHub** : [kestra-io/kestra](https://github.com/kestra-io/kestra)

Kestra peut automatiser et planifier des tâches complexes impliquant PyRIT :
- **CI/CD pour la sécurité de l'IA** : Intégrez PyRIT dans un pipeline Kestra pour exécuter automatiquement des suites de tests de sécurité (par exemple, `XPIATestWorkflow`) à chaque mise à jour d'un modèle ou d'une application.
- **Workflows de Red Teaming planifiés** : Programmez des campagnes de tests récurrentes pour surveiller en continu la posture de sécurité de vos LLMs.
- **Rapports automatisés** : Enchaînez l'exécution de PyRIT avec des étapes de génération et de distribution de rapports de sécurité.

## 3. Surveillance et Alerting avec Prometheus et Alertmanager

**Prometheus** est un système de surveillance et d'alerting open-source, tandis qu'**Alertmanager** gère les alertes envoyées par Prometheus.

- **Dépôts GitHub** :
  - grafana/prometheus (fork populaire)
  - prometheus/alertmanager

L'intégration avec PyRIT permet de surveiller le comportement des systèmes d'IA en production ou en pré-production :
- **Surveillance des performances** : Suivez des métriques comme le temps de réponse des `PromptTarget`, le taux d'échec des requêtes, ou l'utilisation des ressources par les modèles.
- **Détection d'anomalies** : Configurez des alertes avec Alertmanager pour être notifié si un `Scorer` détecte une augmentation soudaine de réponses nuisibles, de refus, ou de scores de vulnérabilité élevés.
- **Corrélation d'événements** : Mettez en corrélation les alertes de sécurité de PyRIT avec des métriques de performance système pour identifier l'impact des attaques ou des défaillances.

## 4. Conformité avec la Directive NIS2

La directive **NIS2 (Network and Information Security 2)** impose des exigences de cybersécurité plus strictes, notamment en matière de gestion des risques de la chaîne d'approvisionnement. L'utilisation combinée de PyRIT et des outils "on-site" aide à répondre à ces exigences.

- **Sécurité de la Chaîne d'Approvisionnement Logicielle** :
  - **Vérification des Commits** : Comme démontré dans le test d'intégration de l'AI Recruiter, la vérification des signatures GPG des commits (`git verify-commit`) garantit l'authenticité et l'intégrité du code source utilisé. L'orchestration avec **Kestra** peut imposer cette vérification comme une étape obligatoire dans les pipelines de CI/CD.
  - **Gestion des Dépendances** : L'utilisation de conteneurs Docker et de versions de commit figées permet de contrôler précisément l'environnement d'exécution, réduisant les risques liés aux dépendances.

- **Gestion des Incidents et Continuité d'Activité** :
  - **Sauvegardes Résilientes** : **Kopia** permet de mettre en place une stratégie de sauvegarde robuste pour les données critiques (comme la mémoire de PyRIT), avec des snapshots immuables et chiffrés, essentiels pour la reprise après incident.
  - **Surveillance et Réponse** : **Prometheus** et **Alertmanager** fournissent les capacités de surveillance en temps réel et d'alerte nécessaires pour détecter rapidement les incidents de sécurité (par exemple, une attaque détectée par PyRIT) et déclencher une réponse.