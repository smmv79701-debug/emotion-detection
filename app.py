import re
import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Emotion Detection", page_icon="😊", layout="centered")

# -------------------------
# File paths
# -------------------------
MODEL_FILE = "bilstm_emotion_model.h5"
TOKENIZER_FILE = "tokenizer(1).pkl"
LABEL_ENCODER_FILE = "label_encoder.pkl"
MAX_LEN_FILE = "max_len.pkl"

# -------------------------
# Text cleaning
# -------------------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -------------------------
# Load saved objects
# -------------------------
@st.cache_resource
def load_all():
    model = load_model(MODEL_FILE)

    with open(TOKENIZER_FILE, "rb") as f:
        tokenizer = pickle.load(f)

    with open(LABEL_ENCODER_FILE, "rb") as f:
        label_encoder = pickle.load(f)

    try:
        with open(MAX_LEN_FILE, "rb") as f:
            max_len = pickle.load(f)
    except FileNotFoundError:
        max_len = 50

    return model, tokenizer, label_encoder, max_len

# -------------------------
# Prediction
# -------------------------
def predict_emotion(text, model, tokenizer, label_encoder, max_len):
    cleaned = clean_text(text)

    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")

    pred = model.predict(padded, verbose=0)[0]
    pred_index = int(np.argmax(pred))
    pred_label = label_encoder.inverse_transform([pred_index])[0]
    confidence = float(np.max(pred))

    probs = {
        label_encoder.inverse_transform([i])[0]: float(pred[i])
        for i in range(len(pred))
    }

    return pred_label, confidence, probs, cleaned

# -------------------------
# UI
# -------------------------
st.title("Emotion Detection from Text 😄😡😢")
st.write("This app predicts emotion using a trained **BiLSTM** model.")

st.markdown("### Supported classes")
st.write("- Happy")
st.write("- Sad")
st.write("- Angry")
st.write("- Fear")

try:
    model, tokenizer, label_encoder, max_len = load_all()

    user_text = st.text_area(
        "Enter text",
        placeholder="Type something like: I am feeling very happy today"
    )

    if st.button("Predict Emotion"):
        if not user_text.strip():
            st.warning("Please enter some text.")
        else:
            label, confidence, probs, cleaned = predict_emotion(
                user_text, model, tokenizer, label_encoder, max_len
            )

            st.success(f"Predicted Emotion: **{label}**")
            st.info(f"Confidence: **{confidence:.2%}**")

            st.markdown("### Cleaned Text")
            st.write(cleaned)

            st.markdown("### Prediction Probabilities")
            for emotion, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                st.write(f"**{emotion}**: {prob:.4f}")

except Exception as e:
    st.error("Error loading model or files.")
    st.exception(e)
    st.markdown("### Make sure these files are in the same folder:")
    st.code(
        "app.py\n"
        "bilstm_emotion_model.h5\n"
        "tokenizer.pkl\n"
        "label_encoder.pkl\n"
        "max_len.pkl"
    )
