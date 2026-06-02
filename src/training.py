"""
training.py — Augmentation, callbacks et boucle d'entraînement.
[Git labels : eda (augmentation), opti-dl (scheduler/callbacks)]

Regroupe :
  - data_augmentation()  : pipeline d'augmentation (ch. 1.6) pour le Jalon 2
  - make_callbacks()     : EarlyStopping + ReduceLROnPlateau (ch. 1.5.2 / 2.8.3)
"""
from __future__ import annotations


def data_augmentation():
    """Couche d'augmentation Keras (ch. 1.6 : forger l'invariance).

    À insérer en tête de modèle ou à mapper sur le train_ds. Rotations, flips
    et zoom : on force le réseau à apprendre l'invariance à l'orientation et
    à l'échelle, ce qui est crucial pour des photos de feuilles prises au
    smartphone sous des angles variés.
    """
    from tensorflow.keras import layers, models

    return models.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
    ], name="data_augmentation")


def make_callbacks(patience: int = 5, monitor: str = "val_loss"):
    """EarlyStopping + ReduceLROnPlateau (scheduler adaptatif, ch. 2.8.3).

    - EarlyStopping : régularisation temporelle (ch. 1.5.2) — on arrête quand
      la validation cesse de progresser et on restaure les meilleurs poids.
    - ReduceLROnPlateau : baisse le LR quand la perte de val stagne.
    """
    from tensorflow.keras import callbacks

    return [
        callbacks.EarlyStopping(monitor=monitor, patience=patience,
                                restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5,
                                    patience=max(2, patience // 2), min_lr=1e-6),
    ]


def train(model, train_ds, val_ds, epochs: int = 30, callbacks_list=None):
    """Boucle d'entraînement standard. Renvoie l'historique Keras."""
    if callbacks_list is None:
        callbacks_list = make_callbacks()
    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=epochs, callbacks=callbacks_list)
    return history