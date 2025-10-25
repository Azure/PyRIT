# Guide de Déploiement Flexible (On-Site et Cloud)

Ce document explique comment configurer et exécuter PyRIT dans un environnement "on-site", c'est-à-dire sans dépendance à des services cloud externes comme Azure ou OpenAI.

## 1. Vue d'ensemble

Un déploiement "on-site" nécessite de remplacer les composants basés sur le cloud par des alternatives locales :

- **Base de données** : Utiliser une base de données locale comme PostgreSQL, SQLite, ou un serveur SQL Server privé au lieu d'Azure SQL.
- **Modèles de Langage (LLM)** : Utiliser des modèles open-source hébergés localement (par ex. via Ollama, vLLM) au lieu des API OpenAI/Azure.
- **Stockage de fichiers** : Utiliser le système de fichiers local au lieu d'Azure Blob Storage.

## 2. Configuration de la Base de Données

Grâce à la classe `SQLAlchemyMemory`, la configuration de la mémoire est flexible. Par défaut, PyRIT utilisera une base de données **SQLite** locale, ce qui ne nécessite aucune configuration.

Pour utiliser une base de données plus robuste comme **PostgreSQL** (recommandé pour la production), installez d'abord le pilote Python :

```sh
pip install psycopg2-binary
```

Ensuite, définissez la variable d'environnement `DB_CONNECTION_STRING` pour pointer vers votre instance de base de données :

```sh
# Exemple pour PostgreSQL
export DB_CONNECTION_STRING="postgresql+psycopg2://user:password@localhost:5432/mydatabase"

# Exemple pour un SQL Server local
export DB_CONNECTION_STRING="mssql+pyodbc://user:password@your_server/your_db?driver=ODBC+Driver+17+for+SQL+Server"
```

Le programme utilisera automatiquement cette base de données au lieu de SQLite.

## 3. Configuration des Modèles de Langage (LLM) Locaux

Pour remplacer les appels à OpenAI/Azure, vous pouvez utiliser des LLM hébergés localement.

### a. Utiliser un serveur compatible API OpenAI (Ollama, vLLM)

De nombreux outils comme Ollama ou vLLM peuvent servir des modèles open-source (Llama3, Mistral, etc.) via une API compatible avec celle d'OpenAI.

Une fois votre serveur local démarré (par ex. `ollama serve`), configurez `OpenAIChatTarget` pour qu'il pointe vers votre endpoint local. Dans votre code Python :

```python
from pyrit.prompt_target import OpenAIChatTarget

# Assurez-vous que votre serveur local est accessible à cette adresse
local_llm_target = OpenAIChatTarget(
    endpoint="http://localhost:11434/v1/chat/completions",
    api_key="any-string-will-do" # La plupart des serveurs locaux n'exigent pas de clé
)

# Utilisez ce target pour vos scorers et autres composants
# Par exemple, pour le SelfAskTrueFalseScorer dans le test de l'AI Recruiter :
true_false_classifier = SelfAskTrueFalseScorer(chat_target=local_llm_target, ...)
```

### b. Remplacer les services de sécurité

Les composants comme `PromptShieldTarget` sont spécifiques à Azure. Pour une analyse de contenu "on-site", vous pouvez :
- Utiliser un LLM local avec un prompt spécifique pour détecter les contenus malveillants.
- Intégrer une bibliothèque open-source de détection de contenu si disponible.
- Pour de nombreux cas d'usage, ce composant peut être désactivé si le risque est jugé acceptable dans un environnement contrôlé.