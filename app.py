import streamlit as st
from PIL import Image, ImageOps, ImageEnhance
import pytesseract
from huggingface_hub import InferenceClient
import os
# =========================
# CONFIG
# =========================
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(token=HF_TOKEN)
st.set_page_config(
    page_title="AI Homework Copilot",
    layout="wide"
)
st.title("📚 AI Homework Copilot")
st.info(
    "Upload homework → OCR extracts text → AI explains step-by-step"
)
# =========================
# IMAGE PREPROCESSING
# =========================
def preprocess_image(image):
    image = ImageOps.grayscale(image)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    return image
# =========================
# OCR FUNCTION
# =========================
def extract_text(image):
    return pytesseract.image_to_string(
        image,
        config="--psm 6"    )
# =========================
# AI TUTOR FUNCTION
# =========================
def get_tutor_response(question):
    prompt = f"""
You are an expert tutor.
Explain the following homework question
in simple language.
Provide:

1. What the question is asking
2. Step-by-step explanation
3. Final answer (if possible)
4. Study tip

Question:

{question}
"""
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-1B-Instruct",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=500,
        temperature=0.4
    )
    return response.choices[0].message.content
# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "Upload Homework Image",
    type=["jpg", "jpeg", "png"]
)
# =========================
# MAIN PIPELINE
# =========================
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(
        image,
        caption="Uploaded Homework",
        use_container_width=True    )
    if st.button("📄 Extract Text"):
        processed_image = preprocess_image(image)
        text = extract_text(processed_image)
        st.session_state["ocr_text"] = text
        st.subheader("📄 Extracted Text")
        if text.strip():
            st.success("Text detected successfully!")
            st.text_area(
                "OCR Result",
                text,
                height=200
            )
        else:
            st.warning(
                "No readable text detected."
            )
# =========================
# EDIT OCR TEXT
# =========================
if "ocr_text" in st.session_state:
    edited_text = st.text_area(
        "✏️ Edit OCR Text (Optional)",
        st.session_state["ocr_text"],
        height=200
    )
    if st.button("🧠 Explain Homework"):
        if not edited_text.strip():
            st.warning(
                "Please provide a question."
            )
        else:
            with st.spinner(
                "AI Tutor is thinking..."
            ):
                answer = get_tutor_response(
                    edited_text
                )
            st.subheader(
                "🧠 AI Tutor Explanation"
            )
            st.write(answer)
# =========================
# MANUAL MODE
# =========================
st.divider()

st.subheader("⌨️ Manual Homework Question")
manual_question = st.text_area(
    "Paste a question directly"
)
if st.button("🚀 Solve Manual Question"):
    if manual_question.strip():
        with st.spinner(
            "AI Tutor is solving..."
        ):
            answer = get_tutor_response(
                manual_question
            )
        st.write(answer)
