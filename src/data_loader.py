"""
data_loader.py — Chargement et préparation du dataset `beans`.

Source : HuggingFace Datasets (beans) — même contenu que le dataset TFDS,
hébergement plus stable (l'URL storage.googleapis.com/ibeans renvoie 403).
"""
from __future__ import annotations
import numpy as np

CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]
NUM_CLASSES = 3
DEFAULT_IMG_SIZE = 128


# ---------------------------------------------------------------------------
# Objet info minimal — reproduit l'interface tfds.core.DatasetInfo utilisée
# dans le notebook (info.features["label"].names, info.splits[s].num_examples)
# ---------------------------------------------------------------------------

class _LabelFeature:
    def __init__(self, names: list[str]):
        self.names = names


class _Features:
    def __init__(self, names: list[str]):
        self._label = _LabelFeature(names)

    def __getitem__(self, key: str):
        if key == "label":
            return self._label
        raise KeyError(key)


class _SplitInfo:
    def __init__(self, num_examples: int):
        self.num_examples = num_examples


class DatasetInfo:
    """Réplique minimale de tfds.core.DatasetInfo."""
    def __init__(self, n_train: int, n_val: int, n_test: int):
        self.features = _Features(CLASS_NAMES)
        self.splits = {
            "train": _SplitInfo(n_train),
            "validation": _SplitInfo(n_val),
            "test": _SplitInfo(n_test),
        }


# ---------------------------------------------------------------------------
# Helpers privés
# ---------------------------------------------------------------------------

def _load_hf_split(split: str, img_size: int):
    """Charge un split HuggingFace beans → tableaux NumPy (images, labels)."""
    from datasets import load_dataset  # type: ignore[import]

    ds = load_dataset("AI-Lab-Makerere/beans", split=split)

    imgs, labels = [], []
    for ex in ds:
        img = np.array(
            ex["image"].resize((img_size, img_size)), dtype=np.float32
        ) / 255.0
        imgs.append(img)
        labels.append(int(ex["labels"]))

    return np.stack(imgs), np.array(labels, dtype=np.int64)


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------

def load_beans_tf(img_size: int = DEFAULT_IMG_SIZE, batch_size: int = 32):
    """
    Charge les splits train/val/test sous forme de tf.data.Dataset.
    Renvoie (train_ds, val_ds, test_ds, info).
    """
    import tensorflow as tf

    X_tr, y_tr = _load_hf_split("train", img_size)
    X_va, y_va = _load_hf_split("validation", img_size)
    X_te, y_te = _load_hf_split("test", img_size)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
        .shuffle(1000)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_va, y_va))
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((X_te, y_te))
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    info = DatasetInfo(len(X_tr), len(X_va), len(X_te))
    return train_ds, val_ds, test_ds, info


def load_beans_numpy(img_size: int = 64):
    """
    Charge tout le dataset en mémoire sous forme NumPy.
    Renvoie X_train, y_train, X_val, y_val, X_test, y_test.
    X : forme (n, img_size*img_size*3), aplati pour la baseline ML.
    """
    X_train, y_train = _load_hf_split("train", img_size)
    X_val, y_val = _load_hf_split("validation", img_size)
    X_test, y_test = _load_hf_split("test", img_size)

    X_train = X_train.reshape(len(X_train), -1)
    X_val = X_val.reshape(len(X_val), -1)
    X_test = X_test.reshape(len(X_test), -1)

    return X_train, y_train, X_val, y_val, X_test, y_test


def class_distribution(y: np.ndarray) -> dict[str, int]:
    """Distribution des classes."""
    counts = np.bincount(y, minlength=NUM_CLASSES)
    return {name: int(c) for name, c in zip(CLASS_NAMES, counts)}
