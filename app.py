import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model, Model
import plotly.graph_objects as go
import time
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from model_utils import predict
from gradcam_utils import make_gradcam_heatmap, overlay_heatmap

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="wide"
)

# ----------------------------
# Load Models
# ----------------------------
@st.cache_resource
def load_models():
    cnn_model = load_model("cnn_model.keras",compile=False)

    feature_extractor = Model(
        inputs=cnn_model.input,
        outputs=cnn_model.layers[-2].output
    )

    with open("pca.pkl", "rb") as f:
        pca = pickle.load(f)

    with open("voting_model.pkl", "rb") as f:
        voting_clf = pickle.load(f)

    with open("class_names.pkl", "rb") as f:
        class_names = pickle.load(f)

    return cnn_model, feature_extractor, pca, voting_clf, class_names


cnn_model, feature_extractor, pca, voting_clf, class_names = load_models()

# ----------------------------
# Header
# ----------------------------
st.title("🧠 Brain Tumor Detection AI")
st.caption("Upload an MRI scan to predict tumor presence")

# ----------------------------
# Upload
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)

    # ----------------------------
    # Prediction
    # ----------------------------
    with st.spinner("Analyzing MRI..."):
        pred_class, confidence, probs, img_input = predict(
            image, feature_extractor, pca, voting_clf, class_names
        )

    label = class_names[pred_class]

    # ----------------------------
    # Layout: Image | Results
    # ----------------------------
    col1, col2 = st.columns(2)

    # LEFT: Image
    with col1:
        st.subheader("Uploaded MRI")
        st.image(image, use_container_width=True)

    # RIGHT: Prediction + Graph
    with col2:
        st.subheader("Prediction")

        if label == "notumor":
            st.success(f"Prediction: {label}")
        else:
            st.error(f"Prediction: {label}")

        st.metric("Confidence", f"{confidence * 100:.2f}%")

    
        # ----------------------------
        # Animated Vertical Graph
        # ----------------------------
        st.subheader("Class Probabilities")

        # Highlight predicted class
        colors = ["#636EFA"] * len(class_names)
        highlight_color = "#00CC96" if label == "notumor" else "#EF553B"
        colors[pred_class] = highlight_color

        chart_placeholder = st.empty()

        # Reset animation on new upload (make sure you already added this)
        # st.session_state.pop("animated", None)

        if "animated" not in st.session_state:
            st.session_state.animated = True

            steps = 25

            for step in range(steps + 1):
                progress = step / steps
                animated_probs = probs * progress

                fig = go.Figure()

                fig.add_trace(go.Bar(
                    x=class_names,
                    y=animated_probs,
                    marker_color=colors,
                    text=[f"{p*100:.1f}%" for p in animated_probs],
                    textposition='outside'
                ))

                fig.update_layout(
                    template="plotly_dark",
                    yaxis=dict(range=[0, 1], title="Probability"),
                    xaxis=dict(title="Classes"),
                    height=400,
                    margin=dict(l=20, r=20, t=30, b=20)
                )

                chart_placeholder.plotly_chart(fig, use_container_width=True)
                time.sleep(0.02)

        else:
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=class_names,
                y=probs,
                marker_color=colors,
                text=[f"{p*100:.1f}%" for p in probs],
                textposition='outside'
            ))

            fig.update_layout(
                template="plotly_dark",
                yaxis=dict(range=[0, 1], title="Probability"),
                xaxis=dict(title="Classes"),
                height=400,
                margin=dict(l=20, r=20, t=30, b=20)
            )

            chart_placeholder.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # Grad-CAM Section (Full Width)
    # ----------------------------
    st.divider()

    if label != "notumor" and confidence > 0.6:
        st.subheader("Tumor Localization (Grad-CAM)")

        heatmap = make_gradcam_heatmap(img_input, cnn_model)

        img_np = np.array(image)

        # Fix channels
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

        img_np = cv2.resize(img_np, (128, 128))
        result = overlay_heatmap(img_np, heatmap)

        st.image(result, use_container_width=True)

    else:
        if label == "notumor":
            st.info("No tumor detected. Grad-CAM is not shown for negative predictions.")
        else:
            st.info("Confidence too low for visualization.")
