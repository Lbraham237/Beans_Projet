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

> Section obligatoire. À compléter honnêtement — voici un canevas à adapter.

**Outils utilisés :** _(ex. : Claude, ChatGPT, Copilot…)_

**Pour quoi faire :**
- Génération du squelette modulaire du projet et de l'arborescence.
- Aide à la rédaction des docstrings et du présent README.
- _(à compléter : debugging, explication de concepts, etc.)_

**Ce que j'ai vérifié / modifié moi-même :**
- _(ex. : j'ai relu et compris chaque fonction de `src/` ; j'ai ajusté les
  hyperparamètres après mes propres expériences ; j'ai validé les résultats.)_

**Limites rencontrées :**
- _(ex. : le code généré utilisait une API scikit-learn dépréciée que j'ai dû
  corriger ; les hyperparamètres suggérés n'étaient pas optimaux pour mon cas.)_

**Ce que j'ai fait sans IA :**
- _(à compléter — important pour démontrer votre compréhension réelle.)_

---

## 📊 Résultats principaux

_(À compléter après entraînement — tableau comparatif depuis le notebook §7.)_

| Modèle | Accuracy (test) | F1 macro | Paramètres | Temps |
|--------|----------------|----------|------------|-------|
| LogReg régularisée (baseline) | … | … | … | … |
| CNN from scratch | … | … | … | … |
| Transfer learning (MobileNetV2) | … | … | … | … |