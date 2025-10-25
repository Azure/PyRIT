# Guide de Déploiement sur FreeBSD

Ce document décrit les étapes et considérations pour exécuter le projet PyRIT sur FreeBSD, en se concentrant sur les dépendances et la configuration spécifiques à l'OS.

## 1. Prérequis Système

Assurez-vous que votre système FreeBSD est à jour et que `pkg` est configuré.

```sh
sudo pkg update && sudo pkg upgrade
```

## 2. Installation de Python

Installez une version récente de Python :

```sh
sudo pkg install python3
```

Il est fortement recommandé de travailler dans un environnement virtuel :

```sh
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Installation des Dépendances

### 3.1. Pilotes ODBC pour Azure SQL

La connexion à Azure SQL via `pyrit.memory.azure_sql_memory` requiert `pyodbc` et les pilotes ODBC de Microsoft.

1.  **Installer unixODBC** :
    ```sh
    sudo pkg install unixODBC
    ```

2.  **Installer le pilote Microsoft ODBC pour SQL Server** :
    Microsoft ne fournit pas de package officiel pour FreeBSD. La compilation depuis les sources ou l'utilisation de ports/packages maintenus par la communauté est nécessaire. Suivez les instructions spécifiques à la communauté FreeBSD pour l'installation de `msodbcsql`.

3.  **Configurer ODBC** :
    Une fois le pilote installé, vous devrez configurer les fichiers `odbc.ini` (pour les DSN) et `odbcinst.ini` (pour les pilotes) pour qu'ils pointent vers le pilote Microsoft ODBC.

### 3.2. Dépendances Python

Installez les dépendances du projet via `pip`. Certaines librairies avec des extensions C (comme `numpy`, `scipy`, `cryptography`) nécessiteront des outils de compilation.

```sh
sudo pkg install gcc py3-pip rust
pip install -r requirements.txt
```

## 4. Considérations sur le Code

Le code source de PyRIT utilise des abstractions qui le rendent majoritairement compatible.

- **Chemins de Fichiers** : Le code utilise `os.path` ou `pathlib`, ce qui le rend compatible avec les chemins de style Unix de FreeBSD.
- **Multiprocessing** : Le module `torch.multiprocessing` est utilisé dans `attack_manager.py`. Sur FreeBSD (comme sur d'autres systèmes Unix), la méthode de démarrage par défaut est `fork`, qui est efficace mais peut causer des problèmes avec les ressources GPU (CUDA). Si vous utilisez des GPU, il peut être nécessaire de forcer la méthode `spawn` au début de votre application :
  ```python
  import torch.multiprocessing as mp

  if __name__ == '__main__':
      mp.set_start_method('spawn')
      # ... reste de votre application
  ```

## 5. Documentation Complémentaire

Le fichier `ai_recruiter_integration_test.fr.md` a été mis à jour pour inclure des détails spécifiques à FreeBSD, notamment sur la manière dont les tests d'intégration peuvent être adaptés et exécutés dans cet environnement.