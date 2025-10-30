# Guide d'utilisation de Mozilla Llamafile

Ce document explique comment utiliser `llamafile` de Mozilla comme cible de prompt dans PyRIT. `llamafile` permet de distribuer et d'exécuter des LLMs dans un unique fichier exécutable, ce qui simplifie considérablement le déploiement local.

## 1. Qu'est-ce que Llamafile ?

Un `llamafile` est un exécutable qui contient à la fois les poids d'un modèle de langage (LLM) et un serveur web intégré. En l'exécutant, vous pouvez interagir avec le modèle via une interface en ligne de commande, une interface web, ou, plus important pour PyRIT, une API compatible avec celle d'OpenAI.

Cela en fait une excellente alternative à des outils comme Ollama ou vLLM pour des déploiements "on-site" sans dépendances complexes.

## 2. Téléchargement et Exécution

1.  **Téléchargez un `llamafile`** :
    Rendez-vous sur le dépôt GitHub de Mozilla-Ocho pour trouver des modèles pré-packagés. Par exemple, pour télécharger `Llama-3-8B-Instruct.llamafile` :

    ```sh
    wget https://huggingface.co/Mozilla/Llama-3-8B-Instruct-llamafile/resolve/main/Llama-3-8B-Instruct.llamafile
    ```

2.  **Rendez le fichier exécutable** :
    ```sh
    chmod +x Llama-3-8B-Instruct.llamafile
    ```

3.  **Démarrez le serveur API** :
    Exécutez le fichier avec l'option `-s` ou `--server` pour démarrer le serveur compatible OpenAI. Par défaut, il écoutera sur le port `8080`.

    ```sh
    ./Llama-3-8B-Instruct.llamafile --server
    ```

## 3. Intégration avec PyRIT

Une fois le serveur `llamafile` en cours d'exécution, vous pouvez le configurer comme une cible dans PyRIT en utilisant `OpenAIChatTarget`. L'endpoint doit pointer vers l'URL du serveur local.

```python
from pyrit.prompt_target import OpenAIChatTarget

# Configurez la cible pour pointer vers votre serveur llamafile local
llamafile_target = OpenAIChatTarget(
    endpoint="http://localhost:8080/v1/chat/completions",
    api_key="not-needed"  # La clé API n'est pas requise par llamafile
)

# Vous pouvez maintenant utiliser `llamafile_target` avec vos scorers, orquestrateurs, etc.
# Par exemple :
# scorer = SelfAskTrueFalseScorer(chat_target=llamafile_target)
# red_teaming_orchestrator = RedTeamingOrchestrator(attack_strategy=..., prompt_target=llamafile_target)
```

Cette configuration vous permet d'utiliser toute la puissance de PyRIT pour évaluer la sécurité et la robustesse de modèles LLM exécutés localement via `llamafile`, sans aucune dépendance à des services cloud.