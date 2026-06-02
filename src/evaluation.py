"""
evaluation.py — Métriques et diagnostics.
[Git labels : eval-ml, eval-dl]

Outils transverses pour :
  - Jalon 4 : diagnostic biais/variance (courbes d'apprentissage, gap train/val)
  - Jalon 7 : comparaison ML vs DL (tableau + temps de calcul)
"""
from __future__ import annotations

import numpy as np


def diagnose_bias_variance(train_acc: float, val_acc: float,
                           threshold_gap: float = 0.10,
                           threshold_low: float = 0.75) -> str:
    """Verdict textuel biais/variance (ch. 1.4) à partir des accuracies.

    Heuristique simple à commenter dans le rapport :
      - train ET val basses        -> biais élevé (sous-apprentissage)
      - train haute, val basse     -> variance élevée (sur-apprentissage)
      - les deux hautes et proches -> bon compromis
    """
    gap = train_acc - val_acc
    if train_acc < threshold_low and val_acc < threshold_low:
        return ("BIAIS ÉLEVÉ (sous-apprentissage) : le modèle est trop simple "
                "pour capturer la structure des données.")
    if gap > threshold_gap:
        return (f"VARIANCE ÉLEVÉE (sur-apprentissage) : écart train-val = "
                f"{gap:.2%}. Le modèle mémorise le bruit.")
    return "COMPROMIS CORRECT : train et validation élevées et proches."


def confusion(y_true, y_pred, class_names=None):
    """Matrice de confusion + rapport de classification (ch. 1.3 évaluation)."""
    from sklearn.metrics import classification_report, confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    return cm, report


def plot_confusion(cm, class_names, ax=None, title="Matrice de confusion"):
    """Affiche une matrice de confusion (à sauver dans reports/figures)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    return ax


def plot_history(history, ax=None):
    """Trace les courbes d'apprentissage (accuracy/loss train vs val).

    Sert directement au diagnostic biais/variance du Jalon 4.
    """
    import matplotlib.pyplot as plt

    h = history.history if hasattr(history, "history") else history
    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h["accuracy"], label="train")
    ax[0].plot(h["val_accuracy"], label="val")
    ax[0].set_title("Accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(h["loss"], label="train")
    ax[1].plot(h["val_loss"], label="val")
    ax[1].set_title("Loss"); ax[1].set_xlabel("epoch"); ax[1].legend()
    return ax


def comparison_table(rows: list[dict]):
    """Construit un DataFrame récapitulatif ML vs DL (Jalon 7).

    Chaque row : {"modele","test_acc","f1_macro","params","temps_s"}.
    """
    import pandas as pd

    df = pd.DataFrame(rows)
    return df.sort_values("test_acc", ascending=False).reset_index(drop=True)