"""
ml_baseline.py — Baseline Machine Learning classique.
[Git label : ml]

Établit une base de référence AVANT le Deep Learning. On utilise une régression
logistique avec les trois régularisations vues en cours (ch. 1.5) :
  - L2 (Ridge)        : pénalité au carré, compresse sans annuler
  - L1 (Lasso)        : pénalité valeur absolue, met des coefficients à zéro (sparsité)
  - ElasticNet        : hybride L1+L2

Sur des pixels bruts aplatis, cette baseline va sous-performer — ce qui est
PRÉCISÉMENT l'argument du Jalon 7 (comparaison ML vs DL) et du Jalon 4
(biais/variance : modèle linéaire = biais élevé sur données spatiales).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# scikit-learn a fait évoluer l'API de régularisation : avant 1.7 il fallait
# spécifier penalty="elasticnet" ; à partir de 1.8 ce paramètre est déprécié
# et seul l1_ratio compte. On adapte automatiquement.
_SKLEARN_VER = tuple(int(p) for p in sklearn.__version__.split(".")[:2])
_PENALTY_DEPRECATED = _SKLEARN_VER >= (1, 8)


@dataclass
class BaselineResult:
    name: str
    penalty: str
    train_acc: float
    val_acc: float
    test_acc: float = float("nan")
    f1_macro: float = float("nan")
    n_zero_coefs: int = 0
    n_total_coefs: int = 0
    extra: dict = field(default_factory=dict)

    def gap(self) -> float:
        """Écart train-val : un grand écart = variance/surapprentissage."""
        return self.train_acc - self.val_acc


def _make_model(penalty: str, C: float, l1_ratio: float | None):
    """Construit une LogisticRegression avec la régularisation choisie.

    C est l'INVERSE de la force de régularisation (petit C = forte pénalité).

    On passe par le solver `saga` + `l1_ratio` qui exprime tout le continuum :
        l1_ratio = 0.0  -> Ridge (L2 pur)
        l1_ratio = 1.0  -> Lasso (L1 pur)
        0 < l1_ratio < 1 -> ElasticNet
    C'est l'API recommandée par scikit-learn récent et elle reste compatible
    avec les versions antérieures.
    """
    ratio = {"l2": 0.0, "l1": 1.0, "elasticnet": l1_ratio if l1_ratio is not None else 0.5}
    if penalty not in ratio:
        raise ValueError(f"penalty inconnue : {penalty}")
    l1_r = float(ratio[penalty])
    if _PENALTY_DEPRECATED:
        return LogisticRegression(solver="saga", l1_ratio=l1_r, C=C, max_iter=2000)
    return LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=l1_r, C=C, max_iter=2000)


def train_baseline(X_train, y_train, X_val, y_val,
                   penalty: str = "l2", C: float = 1.0,
                   l1_ratio: float | None = None,
                   X_test=None, y_test=None) -> BaselineResult:
    """Entraîne une baseline régularisée et renvoie ses métriques."""
    model = _make_model(penalty, C, l1_ratio)
    model.fit(X_train, y_train)

    train_acc = float(accuracy_score(y_train, model.predict(X_train)))
    val_acc = float(accuracy_score(y_val, model.predict(X_val)))

    coefs = model.coef_
    n_zero = int(np.sum(np.abs(coefs) < 1e-6))
    n_total = int(coefs.size)

    res = BaselineResult(
        name=f"LogReg-{penalty}(C={C})",
        penalty=penalty,
        train_acc=train_acc,
        val_acc=val_acc,
        n_zero_coefs=n_zero,
        n_total_coefs=n_total,
    )

    if X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        res.test_acc = float(accuracy_score(y_test, y_pred))
        res.f1_macro = float(f1_score(y_test, y_pred, average="macro"))

    res.extra["model"] = model
    return res


def compare_regularizations(X_train, y_train, X_val, y_val,
                            C: float = 1.0) -> list[BaselineResult]:
    """Compare L2 / L1 / ElasticNet — tableau récapitulatif pour le Jalon 3.

    Illustre concrètement le cours 1.5 : L1 crée de la sparsité (coefs nuls),
    L2 non. À commenter dans le notebook.
    """
    results = []
    for pen, l1r in [("l2", None), ("l1", None), ("elasticnet", 0.5)]:
        results.append(
            train_baseline(X_train, y_train, X_val, y_val,
                           penalty=pen, C=C, l1_ratio=l1r)
        )
    return results