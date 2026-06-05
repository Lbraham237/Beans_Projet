# 🌱 Beans Disease Classifier

> **Auteurs : MVOGO Abraham & MAAROUFI Abdelhamid** — M2 IA, IPSSI Lyon
> Module : PY-ML-DL — Introduction au Machine Learning & Deep Learning

> Détection automatique de maladies du haricot par photo — Machine Learning & Deep Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://beansprojet-cx7e3tuahngmv6oexwsust.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-red?logo=streamlit)
![Accuracy](https://img.shields.io/badge/Accuracy-86.7%25-brightgreen)

---

## Demo

![Demo dashboard](demo_dashboard.gif)

**[→ Tester l'application en ligne](https://beansprojet-cx7e3tuahngmv6oexwsust.streamlit.app/)**

---

## Résultats

| Modèle | Accuracy | F1 macro |
|--------|----------|----------|
| Logistic Regression (baseline) | 63.3 % | 0.634 |
| CNN from scratch | 79.7 % | 0.795 |
| **Transfer Learning (MobileNetV2)** | **86.7 %** | **0.867** |

Le transfer learning surpasse la baseline de **+23 points** grâce aux features ImageNet pré-entraînées.

---

## Problématique

Diagnostiquer l'état sanitaire d'une feuille de haricot à partir d'une photo smartphone — 3 classes : **tache angulaire**, **rouille** et **saine**. Enjeu concret : outil de diagnostic mobile pour petits exploitants agricoles.

Dataset : [`beans`](https://www.tensorflow.org/datasets/catalog/beans) — Makerere AI Lab (1 295 images)

---

## Stack technique

- **ML classique** : Logistic Regression avec régularisation L1/L2/ElasticNet (scikit-learn)
- **Deep Learning** : CNN from scratch + Transfer Learning MobileNetV2 (TensorFlow/Keras)
- **Inférence** : ONNX Runtime (déploiement sans dépendance TensorFlow)
- **Dashboard** : Streamlit — upload image → prédiction + probabilités en temps réel
- **CI/CD** : GitHub Actions + pytest

---

## Structure

```
beans-project/
├── src/
│   ├── data_loader.py      # chargement dataset + augmentation
│   ├── ml_baseline.py      # baseline LogReg
│   ├── models.py           # CNN + MobileNetV2
│   ├── training.py         # callbacks, scheduler, early stopping
│   └── evaluation.py       # métriques, confusion matrix
├── app/dashboard.py        # dashboard Streamlit
├── models/
│   ├── transfer_beans.onnx # modèle déployé (ONNX)
│   └── transfer_beans.keras
├── notebooks/projet_beans.ipynb
└── tests/test_pipeline.py
```

---

## Lancement local

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app/dashboard.py
```

---

## 🤖 Transparence sur l'usage de l'IA générative

**Outils utilisés :** Claude (Anthropic) via Claude Code (assistant CLI)

**Utilisé pour :**
- Génération du squelette modulaire (`src/`, `app/`, `tests/`) et de l'arborescence du projet
- Aide à l'écriture des fonctions dans `data_loader.py`, `ml_baseline.py`, `models.py`, `training.py`, `evaluation.py`
- Migration du chargement de données de TFDS vers HuggingFace Datasets (erreurs 403 sur `storage.googleapis.com`)
- Rédaction des messages de commit et du README
- Debugging (compatibilité scikit-learn 1.7/1.8, types Pylance)
- Conversion du modèle en ONNX et déploiement Streamlit Cloud

**Ce que nous avons vérifié et fait nous-mêmes :**
- Lu et compris chaque fonction des modules `src/`
- Validé les résultats (accuracy, F1, courbes d'apprentissage)
- Testé le pipeline complet de bout en bout dans le notebook
- Choix et justification du dataset `beans` selon les contraintes du sujet
- Analyse critique des résultats et interprétation des métriques
- Compréhension des concepts : régularisation L1/L2/ElasticNet, biais/variance, vanishing gradient, transfer learning

**Limites rencontrées :**
- Le code généré utilisait `tensorflow-datasets` qui retournait des erreurs 403 — corrigé en migrant vers HuggingFace Datasets
- Certains hyperparamètres suggérés ont nécessité des ajustements après observation des courbes d'apprentissage
- TensorFlow incompatible avec Python 3.14 sur Streamlit Cloud — résolu par conversion ONNX

---

## Auteurs

**MVOGO Abraham** & **MAAROUFI Abdelhamid** — M2 IA, IPSSI Lyon
