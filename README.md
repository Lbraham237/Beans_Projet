# 🌱 Classification de maladies du haricot — Projet ML & Deep Learning

> **Auteurs :** MVOGO Abraham & MAAROUFI Abdelhamid

> **Module :** PY-ML-DL-M1 — Introduction au Machine Learning & Deep Learning
> **Dataset :** [`beans`](https://www.tensorflow.org/datasets/catalog/beans) (Makerere AI Lab, 2020)

---

## 🎯 Problématique

Diagnostiquer automatiquement l'état sanitaire d'une feuille de haricot à partir
d'une photo prise au smartphone, parmi **3 classes** : `angular_leaf_spot`
(tache angulaire), `bean_rust` (rouille) et `healthy` (saine). Enjeu agricole
concret : outil de diagnostic mobile pour petits exploitants.

Le projet compare une **baseline Machine Learning classique** à des approches
**Deep Learning** (CNN from scratch puis transfer learning), conformément à la
progression du cours (Forge Prédictive).

---

## 🗂️ Structure du dépôt

```
beans-project/
├── README.md                  # ce fichier
├── requirements.txt           # dépendances (reproductibilité)
├── .gitignore                 # exclut data/ et modèles (Go/No-Go)
├── .github/workflows/ci.yml   # CI/CD : tests automatiques (bonus)
├── src/                       # CODE MÉTIER MODULAIRE (.py importés)
│   ├── data_loader.py         #   chargement TFDS + version NumPy pour le ML
│   ├── ml_baseline.py         #   LogReg régularisée Ridge/Lasso/ElasticNet
│   ├── models.py              #   MLP, CNN, transfer learning MobileNetV2
│   ├── training.py            #   augmentation + callbacks (scheduler, early stop)
│   └── evaluation.py          #   biais/variance, confusion, comparaison
├── notebooks/
│   └── projet_beans.ipynb     # notebook d'orchestration (importe src/)
├── app/
│   └── dashboard.py           # dashboard Streamlit (Jalon 9)
├── tests/
│   └── test_pipeline.py       # tests unitaires (bonus CI/CD)
└── reports/figures/           # figures générées pour le rapport
```

> Le dossier `data/` et les modèles entraînés **ne sont pas versionnés**
> (voir `.gitignore`). Le dataset se télécharge automatiquement via TFDS.

---

## 🚀 Installation & reproductibilité

```bash
python -m venv .venv && source .venv/bin/activate   # ou: uv venv
pip install -r requirements.txt
```

**Reproduire l'analyse :**
```bash
jupyter notebook notebooks/projet_beans.ipynb   # entraîne et sauve les modèles
streamlit run app/dashboard.py                   # lance le dashboard
pytest -q tests/                                  # lance les tests
```

> Sur Apple Silicon : remplacer `tensorflow` par `tensorflow-macos` +
> `tensorflow-metal` dans `requirements.txt`.

---

## 🏗️ Démarche par jalon

| Partie | Jalon | Où le trouver |
|--------|-------|---------------|
| 1 | Data + EDA + augmentation | `data_loader.py`, `training.py`, notebook §1-2 |
| 1 | Baseline ML + régularisation L1/L2/ElasticNet | `ml_baseline.py`, notebook §3 |
| 1 | Évaluation ML + biais/variance | `evaluation.py`, notebook §4 |
| 2 | Architecture (CNN justifié vs MLP) | `models.py`, notebook §5 |
| 2 | Optimisation (Adam, BN, scheduler, anti-vanishing) | `models.py`, `training.py`, notebook §6 |
| 2 | Comparaison ML vs DL | `evaluation.py`, notebook §7 |
| 3 | DL avancé : transfer learning MobileNetV2 | `models.py`, notebook §8 |
| 3 | Déploiement : dashboard Streamlit | `app/dashboard.py` |
| 3 | CI/CD (bonus) | `.github/workflows/ci.yml`, `tests/` |

---

## 🤖 Transparence sur l'usage de l'IA générative

**Outils utilisés :** Claude (Anthropic) — via Claude Code (assistant CLI)

**Pour quoi faire :**
- Génération du squelette modulaire du projet et de l'arborescence (`src/`, `app/`, `tests/`).
- Aide à l'écriture des fonctions dans `data_loader.py`, `ml_baseline.py`, `models.py`, `training.py`, `evaluation.py`.
- Migration du chargement de données de TFDS vers HuggingFace Datasets suite aux erreurs 403.
- Rédaction des messages de commit et du présent README.
- Debugging (compatibilité scikit-learn 1.7/1.8, types Pylance).

**Ce que nous avons vérifié / modifié nous-mêmes :**
- Relu et compris chaque fonction des modules `src/`.
- Validé les résultats obtenus (accuracy, F1, courbes d'apprentissage).
- Testé le pipeline complet de bout en bout dans le notebook.

**Limites rencontrées :**
- Le code généré utilisait `tensorflow-datasets` qui retournait des erreurs 403 sur `storage.googleapis.com` — corrigé en migrant vers HuggingFace Datasets.
- Certaines suggestions d'hyperparamètres ont nécessité des ajustements après observation des courbes d'apprentissage.

**Ce que nous avons fait sans IA :**
- Choix et justification du dataset `beans` (Makerere AI Lab) selon les contraintes du sujet.
- Compréhension des concepts : régularisation L1/L2/ElasticNet, biais/variance, vanishing gradient, transfer learning.
- Analyse critique des résultats et interprétation des métriques.

---

## 📊 Résultats principaux

| Modèle | Accuracy (test) | F1 macro | Paramètres | Temps |
|--------|----------------|----------|------------|-------|
| LogReg régularisée (baseline) | 63.3 % | 0.634 | 36 864 | rapide |
| CNN from scratch | 79.7 % | 0.795 | 94 307 | moyen |
| Transfer learning (MobileNetV2) | **86.7 %** | **0.867** | 2 261 827 | moyen |

> Le transfer learning surpasse nettement la baseline ML (+23 points), confirmant l'apport des features visuelles pré-entraînées sur ImageNet pour une tâche de classification d'images botaniques.
>
![Demo dashboard](demo_dashboard.gif)
