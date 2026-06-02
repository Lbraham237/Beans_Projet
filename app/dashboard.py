"""
app/dashboard.py — Dashboard interactif (Jalon 9).
[Git label : deploy]

Permet à un utilisateur d'uploader une photo de feuille de haricot et d'obtenir
la prédiction du modèle (sain / tache angulaire / rouille), avec les probabilités.

Lancement :  streamlit run app/dashboard.py
Prérequis  :  un modèle entraîné sauvegardé dans models/cnn_beans.keras
              (généré par le notebook). Le modèle n'est PAS versionné (.gitignore).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_loader import CLASS_NAMES, DEFAULT_IMG_SIZE  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "cnn_beans.keras"
LABEL_FR = {
    "angular_leaf_spot": "Tache angulaire 🦠",
    "bean_rust": "Rouille du haricot 🍂",
    "healthy": "Feuille saine ✅",
}


@st.cache_resource
def load_model():
    """Charge le modèle entraîné (mis en cache entre les interactions)."""
    from tensorflow.keras.models import load_model as _load
    if not MODEL_PATH.exists():
        return None
    return _load(MODEL_PATH)


def preprocess(pil_image, img_size: int = DEFAULT_IMG_SIZE) -> np.ndarray:
    """Prépare l'image uploadée comme à l'entraînement : resize + normalisation."""
    img = pil_image.convert("RGB").resize((img_size, img_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]  # ajout dimension batch


def main():
    st.set_page_config(page_title="Beans Disease Classifier", page_icon="🌱")
    st.title("🌱 Détection de maladies du haricot")
    st.caption("Projet ML/DL — classification de feuilles (dataset beans, Makerere AI Lab)")

    model = load_model()
    if model is None:
        st.warning(
            "Aucun modèle trouvé. Entraînez d'abord le CNN dans le notebook "
            "et sauvegardez-le dans `models/cnn_beans.keras`."
        )
        st.stop()

    uploaded = st.file_uploader("Choisissez une photo de feuille",
                                type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Uploadez une image pour lancer la prédiction.")
        return

    from PIL import Image
    image = Image.open(uploaded)
    st.image(image, caption="Image fournie", use_container_width=True)

    x = preprocess(image)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    pred = CLASS_NAMES[idx]

    st.subheader(f"Prédiction : {LABEL_FR.get(pred, pred)}")
    st.metric("Confiance", f"{probs[idx]:.1%}")

    st.write("**Détail des probabilités :**")
    for name, p in zip(CLASS_NAMES, probs):
        st.progress(float(p), text=f"{LABEL_FR.get(name, name)} — {p:.1%}")


if __name__ == "__main__":
    main()