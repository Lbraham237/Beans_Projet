"""
models.py — Architectures Deep Learning.
[Git labels : dl, opti-dl]

Trois constructeurs de modèles, du plus simple au plus avancé :
  1. build_mlp_baseline  — MLP naïf (montre l'explosion du nombre de poids, ch. 2.8)
  2. build_cnn           — CNN from scratch (Jalon 5 : localité + partage de poids)
  3. build_transfer      — Transfer learning MobileNetV2 (Jalon 8 : techno de pointe)

Le CNN applique les bonnes pratiques anti-vanishing-gradient du cours :
ReLU + BatchNormalization placée AVANT l'activation (ordre Linear→BN→Activation,
cf. ch. 2.7.1), initialisation He, et Adam comme optimiseur (ch. 1.3.4).
"""
from __future__ import annotations


def build_mlp_baseline(img_size: int = 128, num_classes: int = 3):
    """MLP totalement connecté — sert à ILLUSTRER ses limites (ch. 2.8).

    Sur 128x128x3 la première couche dense explose en paramètres : c'est
    l'argument qui justifie le CNN au Jalon 5.
    """
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input((img_size, img_size, 3)),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ], name="mlp_baseline")
    return model


def build_cnn(img_size: int = 128, num_classes: int = 3,
              use_batchnorm: bool = True):
    """CNN from scratch — Jalon 5.

    Bloc type : Conv -> BN -> ReLU -> MaxPool, profondeur croissante 32->64->128.
    GlobalAveragePooling (ch. 4.2) au lieu de Flatten+Dense géant : régularisateur
    naturel et bien moins de paramètres.
    """
    from tensorflow.keras import layers, models

    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same",
                          kernel_initializer="he_normal", use_bias=not use_batchnorm)(x)
        if use_batchnorm:
            x = layers.BatchNormalization()(x)   # ordre Linear -> BN -> Activation
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        return x

    inputs = layers.Input((img_size, img_size, 3))
    x = conv_block(inputs, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="cnn_from_scratch")


def build_transfer(img_size: int = 128, num_classes: int = 3,
                   trainable_base: bool = False):
    """Transfer learning avec MobileNetV2 pré-entraîné ImageNet — Jalon 8.

    On gèle le backbone (trainable_base=False) pour un fine-tuning rapide,
    adapté à "quelques jours" et au petit dataset beans.
    """
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2  # type: ignore[import]

    base = MobileNetV2(input_shape=(img_size, img_size, 3),
                       include_top=False, weights="imagenet")
    base.trainable = trainable_base

    inputs = layers.Input((img_size, img_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="transfer_mobilenetv2")


def compile_model(model, lr: float = 1e-3):
    """Compile avec Adam (ch. 1.3.4) et cross-entropy (labels entiers)."""
    from tensorflow.keras.optimizers import Adam

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model