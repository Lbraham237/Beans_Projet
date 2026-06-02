"""
data_loader.py — Chargement et préparation du dataset `beans`.
"""

from __future__ import annotations
import numpy as np

CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]
NUM_CLASSES = 3
DEFAULT_IMG_SIZE = 128  # redimensionnement des images


def load_beans_tf(img_size: int = DEFAULT_IMG_SIZE, batch_size: int = 32):
    """
    Charge les splits train/val/test sous forme de tf.data.Dataset.
    Renvoie (train_ds, val_ds, test_ds, info).
    """
    import tensorflow as tf
    import tensorflow_datasets as tfds

    (train_ds, val_ds, test_ds), info = tfds.load(  # type: ignore[misc]
        "beans",
        split=["train", "validation", "test"],
        as_supervised=True,
        with_info=True,
        shuffle_files=True,
    )

    def _prep(image, label):
        image = tf.image.resize(image, (img_size, img_size))
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = (
        train_ds.map(_prep, num_parallel_calls=AUTOTUNE)
        .shuffle(1000)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        val_ds.map(_prep, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    test_ds = (
        test_ds.map(_prep, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds, test_ds, info


def load_beans_numpy(img_size: int = 64):
    """
    Charge tout le dataset en mémoire sous forme NumPy.
    Renvoie X_train, y_train, X_val, y_val, X_test, y_test.
    """
    import tensorflow as tf
    import tensorflow_datasets as tfds

    def _to_numpy(split):
        ds = tfds.load("beans", split=split, as_supervised=True)
        xs, ys = [], []

        for image, label in tfds.as_numpy(ds):
            image = tf.image.resize(image, (img_size, img_size)).numpy()
            xs.append(image.astype(np.float32) / 255.0)
            ys.append(int(label))

        X = np.stack(xs).reshape(len(xs), -1)
        y = np.array(ys)
        return X, y

    X_train, y_train = _to_numpy("train")
    X_val, y_val = _to_numpy("validation")
    X_test, y_test = _to_numpy("test")

    return X_train, y_train, X_val, y_val, X_test, y_test


def class_distribution(y: np.ndarray) -> dict[str, int]:
    """Distribution des classes."""
    counts = np.bincount(y, minlength=NUM_CLASSES)
    return {name: int(c) for name, c in zip(CLASS_NAMES, counts)}
