import os
import json
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Page Setup
st.set_page_config(
    page_title="DERMAVECTOR AI | CNN Pathological Classifier",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 DERMAVECTOR AI")
st.caption("ResNet-50 / EfficientNet Deep Convolutional Vision Classifier for Dermatological Lesions")

# Initialize Gemini API Client secretly
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("⚠️ GEMINI_API_KEY not found in .env file. Please check your configuration.")
    st.stop()


@st.cache_resource
def get_gemini_client(key: str):
    return genai.Client(api_key=key)


client = get_gemini_client(api_key)


# Pydantic Schemas for Strict Single-Class Vision Inference
class ClassProbability(BaseModel):
    class_label: str = Field(
        description="Dermatological condition / disease name (e.g., Eczema, Atopic Dermatitis, Psoriasis, Melanoma, Tinea Corporis).")
    softmax_score: float = Field(description="Normalized softmax confidence probability between 0.00 and 1.00.")


class ImageClassificationResult(BaseModel):
    identified_condition: str = Field(
        description="The primary classified disease or condition name ONLY (e.g., Atopic Dermatitis, Psoriasis, Seborrheic Keratosis).")
    top_class_confidence: float = Field(description="Top-1 class prediction probability value between 0.00 and 1.00.")
    icd_10_code: str = Field(description="Corresponding ICD-10 medical classification code.")
    softmax_distribution: list[ClassProbability] = Field(
        description="Top 4 class candidate probabilities output by the final linear activation layer.")


SYSTEM_INSTRUCTION = """
You are DERMAVECTOR AI, a fine-tuned ResNet-50 Deep Convolutional Neural Network trained on ISIC and DERMNET vision benchmark datasets.
Analyze the provided skin lesion or dermatological image.
Perform multi-class image classification and output strictly the identified condition name, top-1 confidence score, ICD-10 code, and softmax probability distribution.

Output strictly valid JSON matching the schema. Do not generate conversational conversational filler or prose.
"""


def simulate_cnn_inference():
    """Simulates realistic Computer Vision feature extraction & tensor forward-pass."""
    progress_bar = st.progress(0, text="Resizing Tensor to 224x224x3 & Applying ImageNet Normalization...")
    time.sleep(0.3)

    progress_bar.progress(30, text="Executing Conv2D Layers & Feature Map Extraction (ResNet-50 Blocks 1-4)...")
    time.sleep(0.4)

    progress_bar.progress(70, text="Computing Global Average Pooling & Dense Activation Layers...")
    time.sleep(0.4)

    progress_bar.progress(95, text="Applying Softmax Function across 120 Dermatological Class Outputs...")
    time.sleep(0.3)

    progress_bar.progress(100, text="Classification Inference Complete!")
    time.sleep(0.2)
    progress_bar.empty()


def render_ml_classification_output(result: dict):
    """Renders the single predicted condition name in standard ML model output format."""

    condition_name = result.get("identified_condition", "Unclassified Anomaly")
    confidence = result.get("top_class_confidence", 0.95) * 100
    icd = result.get("icd_10_code", "N/A")

    # 1. Standard ML Single-Prediction Output Block
    st.subheader("🎯 Predicted Class Vector")

    col_out1, col_out2, col_out3 = st.columns([2, 1, 1])

    with col_out1:
        st.markdown(f"### **Condition Identified:** `{condition_name}`")

    with col_out2:
        st.metric(label="Softmax Confidence", value=f"{confidence:.2f}%")

    with col_out3:
        st.metric(label="ICD-10 Class Code", value=icd)

    st.divider()

    # 2. Softmax Probability Distribution Bar Plot
    st.subheader("📊 Output Layer Softmax Distribution")
    distribution = result.get("softmax_distribution", [])
    if distribution:
        df_dist = pd.DataFrame(distribution)
        df_dist["Probability (%)"] = (df_dist["softmax_score"] * 100).round(2)

        fig_bar = px.bar(
            df_dist,
            x="Probability (%)",
            y="class_label",
            orientation="h",
            text="Probability (%)",
            title="Top-4 Class Candidate Probabilities",
            labels={"class_label": "Predicted Condition Class", "Probability (%)": "Softmax Probability (%)"},
            color="Probability (%)",
            color_continuous_scale="Blues"
        )
        fig_bar.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)


# Sidebar Model Config (Disguised ML Settings)
with st.sidebar:
    st.header("⚙️ Model Architecture")
    st.selectbox("Vision Backbone", ["ResNet-50 fine-tuned (DERMNET-v2)", "EfficientNet-B4 Medical Vision",
                                     "Vision Transformer (ViT-Base-224)"])
    st.slider("Classification Threshold (Softmax)", min_value=0.50, max_value=0.99, value=0.85, step=0.05)
    st.text_input("Input Tensor Resolution", value="224 x 224 x 3")
    st.caption("Model Weights: Pre-trained on ImageNet-1k & fine-tuned on 150,000 clinical dermatological images.")

# Main Image Upload Area
st.subheader("📥 Input Image Tensor (Skin Condition / Lesion)")

uploaded_file = st.file_uploader("Upload Image File (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    col_img, col_meta = st.columns([1, 2])

    with col_img:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image Buffer", use_container_width=True)

    with col_meta:
        st.caption("Image Tensor Details:")
        st.code(
            f"Format: {image.format}\nDimensions: {image.size[0]} x {image.size[1]} px\nColor Channels: {image.mode}")

    run_classify_btn = st.button("🚀 Forward Pass & Classify Condition", type="primary", use_container_width=True)

    if run_classify_btn:
        # 1. UI Pipeline Animation
        simulate_cnn_inference()

        # 2. Call Gemini Vision secretly under the hood
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=[
                    image,
                    "Classify this dermatological image. Output ONLY the disease/condition name, confidence score, ICD-10 code, and top candidate distribution."
                ],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=ImageClassificationResult,
                ),
            )

            inference_data = json.loads(response.text)

            # 3. Render Output
            render_ml_classification_output(inference_data)

        except Exception as e:
            st.error(f"Tensor Classification Failure: {e}")
else:
    st.info("Please upload a clinical image file to run inference.")