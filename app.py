import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import joblib

from utils import load_image, extract_features
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Innovation in Stroke Identification",
    page_icon="🧠",
    layout="wide"
)

# ================= BLACK PANTHER THEME =================
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #2b0033, #000000 70%);
    color: #ffffff;
    font-family: 'Segoe UI', sans-serif;
}

.block-container {
    background: rgba(8, 8, 15, 0.95);
    border-radius: 22px;
    padding: 35px;
    box-shadow:
        0 0 40px rgba(187, 134, 252, 0.25),
        inset 0 0 20px rgba(187, 134, 252, 0.08);
    margin-top: 30px;
}

h1 {
    text-align: center;
    font-weight: 900;
    letter-spacing: 1.5px;
    background: linear-gradient(90deg, #c77dff, #9b59ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

h2, h3 {
    text-align: center;
    color: #e0c3ff;
}

[data-testid="stFileUploader"] {
    background: #0f0f1a;
    border: 1px solid #bb86fc;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 0 18px rgba(187, 134, 252, 0.3);
}

img {
    border-radius: 16px;
    box-shadow: 0 0 30px rgba(187, 134, 252, 0.35);
}

.result-normal {
    background: linear-gradient(135deg, #003300, #00ff99);
    color: black;
    padding: 25px;
    border-radius: 16px;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
}

.result-stroke {
    background: linear-gradient(135deg, #660000, #ff1744);
    color: white;
    padding: 25px;
    border-radius: 16px;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ================= TITLE (UNCHANGED) =================
st.title("🧠 Innovation in Stroke Identification")
st.subheader("ML-Based Diagnostic Model Using Neuroimages")

# ================= STRICT BRAIN CT VALIDATION =================
def is_brain_ct_image(image):
    img = np.array(image)

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    h, w = gray.shape
    score = 0

    # Aspect ratio
    if 0.65 <= (w / h) <= 1.35:
        score += 1

    # Skull brightness
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    white_ratio = np.sum(thresh > 0) / thresh.size
    if 0.10 <= white_ratio <= 0.75:
        score += 1

    # Symmetry
    left = gray[:, :w//2]
    right = cv2.flip(gray[:, w//2:], 1)
    min_w = min(left.shape[1], right.shape[1])
    left_crop = left[:, :min_w]
    right_crop = right[:, :min_w]
    diff = np.mean(np.abs(left_crop - right_crop))

    if diff < 60:
        score += 1

    # Edge density
    edges = cv2.Canny(gray, 40, 140)
    edge_ratio = np.sum(edges > 0) / edges.size
    if 0.005 <= edge_ratio <= 0.25:
        score += 1

    # Long straight bone (leg rejection)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, 120,
        minLineLength=int(0.6 * min(h, w)),
        maxLineGap=10
    )
    if lines is not None:
        return False
    # 6️⃣ Brain CT intensity range check (STRICT)
    mean_intensity = np.mean(gray)
    if not (40 <= mean_intensity <= 110):
        return False

    return score >= 3

# ================= LOAD MODELS =================
@st.cache_resource
def load_all():
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3)
    )
    feature_model = Model(inputs=base_model.input, outputs=base_model.output)

    bilstm_model = tf.keras.models.load_model("ga_bilstm_stroke_model.h5")
    mask = np.load("selected_mask.npy")
    scaler = joblib.load("feature_scaler.pkl")

    return feature_model, bilstm_model, mask, scaler

feature_model, bilstm_model, mask, scaler = load_all()

# ================= FILE UPLOAD =================
uploaded_file = st.file_uploader(
    "Upload Brain CT Scan (PNG / JPG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    status = st.empty()
    status.info("🧠 AI is analyzing the CT scan...")

    image = load_image(uploaded_file)

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.image(image, caption="Uploaded Brain CT Image", width=350)

    # 🔴 Image-level validation
    if not is_brain_ct_image(image):
        status.error("❌ Invalid Image")
        st.warning("Only valid BRAIN CT scan images are accepted.")
        st.stop()

    # 🔹 Resize explicitly for ResNet50
    image_resized = cv2.resize(image, (224, 224))

    # Feature extraction
    features = extract_features(image_resized, feature_model)

    # Feature scaling & selection
    features = scaler.transform(features)
    features = features[:, mask]
    features = features.reshape((features.shape[0], features.shape[1], 1))

    # Prediction
    prediction = bilstm_model.predict(features, verbose=0)[0][0]
    status.success("✅ CT scan analyzed successfully")

    confidence = prediction if prediction >= 0.5 else 1 - prediction
    st.write(f"🧪 Raw Model Prediction: **{prediction:.4f}**")

    if prediction >= 0.5:
        st.markdown(
            f"<div class='result-stroke'>🛑 Stroke Detected<br>Confidence: {confidence*100:.2f}%</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result-normal'>✅ Normal Brain Scan<br>Confidence: {confidence*100:.2f}%</div>",
            unsafe_allow_html=True
        )
