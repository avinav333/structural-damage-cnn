"""
Streamlit Live Demo — Structural Damage Detection
Upload a concrete surface image and get instant crack detection results.

Run locally:  streamlit run app.py
Deploy free:  https://streamlit.io/cloud
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
import os

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Structural Damage Detector",
    page_icon="🏗️",
    layout="centered"
)

# ── Header ────────────────────────────────────────────────
st.title("🏗️ Structural Damage Detection")
st.markdown("""
**AI-powered crack detection on concrete surfaces from drone imagery.**  
Upload a concrete surface image to detect structural damage instantly.

> *Project by Abhinava Mondal — B.Tech Construction Engineering, Jadavpur University*
""")

st.divider()

# ── Load model (with fallback demo mode) ──────────────────
@st.cache_resource
def load_model():
    try:
        import tensorflow as tf
        if os.path.exists("damage_model.h5"):
            model = tf.keras.models.load_model("damage_model.h5")
            return model, "real"
        else:
            return None, "demo"
    except ImportError:
        return None, "demo"

model, mode = load_model()

if mode == "demo":
    st.info("⚠️ Running in **Demo Mode** — model not loaded. Upload any image to see the interface. "
            "To run with the real model, train it first using `train.py` and place `damage_model.h5` here.")

# ── Image upload ──────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a concrete surface image",
    type=["jpg", "jpeg", "png"],
    help="Best results with close-up images of concrete walls, floors, or bridges."
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(image, use_column_width=True)

    # ── Prediction ────────────────────────────────────────
    IMG_SIZE = (128, 128)

    if mode == "real" and model is not None:
        import tensorflow as tf
        img_array = np.array(image.resize(IMG_SIZE)) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prob = float(model.predict(img_array, verbose=0)[0][0])
        label = "Cracked" if prob > 0.5 else "Not Cracked"
        confidence = prob if prob > 0.5 else 1 - prob
    else:
        # Demo mode: simulate prediction based on image brightness
        gray = np.array(image.convert("L"))
        variance = float(np.var(gray))
        # Higher variance → more likely cracked (rough heuristic for demo)
        prob = min(0.95, max(0.05, variance / 3000))
        label = "Cracked" if prob > 0.5 else "Not Cracked"
        confidence = prob if prob > 0.5 else 1 - prob

    with col2:
        st.subheader("🔍 Detection Result")
        if label == "Cracked":
            st.error(f"### ⚠️ {label}")
            st.metric("Damage Confidence", f"{confidence*100:.1f}%")
            st.markdown("""
**Recommendation:**
- Flag for structural inspection
- Schedule engineering assessment
- Do not proceed with loading until cleared
""")
        else:
            st.success(f"### ✅ {label}")
            st.metric("Confidence", f"{confidence*100:.1f}%")
            st.markdown("""
**Status:**
- No significant surface damage detected
- Continue routine monitoring schedule
""")

        # Confidence bar
        st.progress(confidence)

    # ── Heatmap overlay (OpenCV highlight) ────────────────
    st.divider()
    st.subheader("🗺️ Damage Localisation")

    try:
        import cv2
        img_cv = np.array(image.resize((300, 300)))
        gray_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        # Canny edge detection to highlight potential crack regions
        edges = cv2.Canny(gray_cv, threshold1=50, threshold2=150)
        edges_colored = cv2.applyColorMap(edges, cv2.COLORMAP_HOT)
        overlay = cv2.addWeighted(cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR), 0.6,
                                  edges_colored, 0.4, 0)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        st.image(overlay_rgb, caption="Edge-highlighted overlay (potential damage zones in red/yellow)", use_column_width=True)
    except ImportError:
        st.info("Install opencv-python for damage localisation overlay.")

    st.divider()
    st.caption("Model: Lightweight CNN trained on SDNET2018 dataset | "
               "Abhinava Mondal, Jadavpur University | Mission Sudarshan Chakra — Atmanirbhar Bharat")

else:
    # Show sample instructions
    st.markdown("""
### How to use
1. 📁 Upload a **JPG or PNG** image of a concrete surface
2. 🤖 The AI model analyses the image for cracks and spalling
3. 📊 View the **damage classification** and **confidence score**
4. 🗺️ See the **damage localisation overlay** highlighting affected regions

### Example use cases
- Post-disaster rapid structural assessment
- Drone-based bridge and building inspection
- Construction site quality monitoring
- Military infrastructure damage assessment
""")

    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Concrete_cracks.jpg/640px-Concrete_cracks.jpg",
        caption="Example: Concrete surface with visible crack damage",
        use_column_width=True
    )
