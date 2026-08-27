# ARC-AGI-3 (Kaggle: arc-prize-2026-arc-agi-3)

Compétition d'**agents interactifs** : on ne prédit pas des grilles statiques,
on fait jouer un agent à des jeux (`environment_files/`) via le package `arc_agi`.
Deux modes :

- **OFFLINE** : jeux exécutés localement, aucun accès réseau ni clé API. C'est le
  mode utilisé par `notebooks/starter.ipynb` et probablement le mode dans lequel
  tourne l'évaluation finale du notebook soumis (pas d'accès internet garanti).
- **ONLINE/NORMAL** : joue contre le serveur `three.arcprize.org`, nécessite une
  clé `ARC_API_KEY` (compte séparé sur https://three.arcprize.org/, différent de Kaggle).

## Environnement

- Python 3.12.13 (pyenv, pin via `.python-version` — requis par les wheels
  précompilées que Kaggle fournit dans `data/arc_agi_3_wheels/`, taguées cp312)
- venv local dans `.venv/`
- Kernel Jupyter enregistré : `arc-agi-3`

```bash
source .venv/bin/activate
jupyter notebook notebooks/starter.ipynb
```

## Kaggle API (téléchargement des données de la compétition)

Token API déjà configuré dans `~/.kaggle/access_token`. Pour retélécharger les données :

```bash
source .venv/bin/activate
kaggle competitions download -c arc-prize-2026-arc-agi-3 -p data
unzip -o data/arc-prize-2026-arc-agi-3.zip -d data && rm data/arc-prize-2026-arc-agi-3.zip
```

Le zip contient :
- `data/ARC-AGI-3-Agents/` — framework d'agents complet (templates random,
  LangGraph, smolagents...), avec son propre `.env` (mode offline configuré).
- `data/environment_files/` — les jeux jouables localement (25 environnements).
- `data/arc_agi_3_wheels/` — wheels Python 3.12 pour installer `arc_agi`/`arcengine`
  et leurs dépendances sans accès internet (utile si le notebook Kaggle tourne
  sans internet lors de la soumission).

## ARC_API_KEY (mode online, optionnel)

1. Créer un compte sur https://three.arcprize.org/ et récupérer une clé API.
2. La mettre dans `data/ARC-AGI-3-Agents/.env` (`ARC_API_KEY=...`) et passer
   `OPERATION_MODE=normal` ou `online` pour jouer contre le serveur officiel.

## Lancer un agent (framework complet, hors notebook)

```bash
cd data/ARC-AGI-3-Agents
source ../../.venv/bin/activate
python main.py --agent=random --game=ls20   # nécessite ARC_API_KEY (voir note ci-dessous)
```

Note : `main.py` résout toujours la liste des jeux via l'API en ligne avant de
jouer, même en mode offline — une clé `ARC_API_KEY` valide est donc nécessaire
pour l'utiliser tel quel. Pour un usage 100% offline (comme dans le notebook),
utiliser directement `arc_agi.Arcade(operation_mode="offline")` sans passer par
`main.py`.

## Soumission

Le notebook `notebooks/starter.ipynb` détecte automatiquement s'il tourne sur Kaggle
(`/kaggle/input/...`) ou en local (`data/`), et joue en mode offline (aucune
dépendance réseau). Pour pousser une version sur Kaggle :

```bash
kaggle kernels push -p notebooks/
```

(nécessite un `kernel-metadata.json` dans `notebooks/` — généré via
`kaggle kernels init -p notebooks/` si besoin.)

## Structure

```
.
├── data/                  # données de la compétition (gitignore, à télécharger)
│   ├── ARC-AGI-3-Agents/  # framework d'agents fourni par Kaggle
│   ├── environment_files/ # jeux locaux
│   └── arc_agi_3_wheels/  # wheels offline (Python 3.12)
├── notebooks/             # notebooks, dont celui à soumettre
├── src/                   # code partagé importable depuis les notebooks
├── requirements.txt
└── .venv/                 # environnement virtuel (gitignore)
```
