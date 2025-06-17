# Breast Cancer Chatbot Web App - Hybrid (File + General Knowledge)
# Streamlit Web Version using OpenRouter API and DeepSeek

import streamlit as st
import fitz  # PyMuPDF
import json
import requests
import tempfile
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(page_title="Breast Cancer Chatbot", layout="wide")
st.title("🩺 Breast Cancer Chatbot (Hybrid Knowledge)")
st.markdown("Ask medical questions and get answers based on clinical guidelines and general medical knowledge.")

# Sidebar API Key and Model
with st.sidebar:
    st.header("🔑 OpenRouter API")
    api_key = st.text_input("Enter OpenRouter API key", type="password")
    model = st.text_input("Model name", value="deepseek/deepseek-r1:free")

# File upload section
uploaded_file = st.file_uploader("Upload a PDF of Breast Cancer Guidelines", type="pdf")
pdf_text = ""

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with fitz.open(tmp_path) as doc:
        for page in doc:
            pdf_text += page.get_text()

    # Split into chunks
    def split_into_chunks(text, chunk_size=500, overlap=100):
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i + chunk_size])
        return chunks

    chunks = split_into_chunks(pdf_text)
    vectorizer = TfidfVectorizer().fit(chunks)
    chunk_vectors = vectorizer.transform(chunks)

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Chat interface
    user_input = st.text_input("Ask your medical question:")
    if user_input and api_key:
        # Retrieve top 3 chunks
        query_vec = vectorizer.transform([user_input])
        sims = cosine_similarity(query_vec, chunk_vectors).flatten()
        top_indices = sims.argsort()[-3:][::-1]
        top_chunks = "\n\n".join([chunks[i] for i in top_indices])

        messages = [
            {"role": "system", "content": "You are a helpful medical assistant. Use provided context if available, otherwise answer from medical knowledge."},
            {"role": "user", "content": f"Context from file:\n{top_chunks}\n\nQuestion: {user_input}"}
        ]

        # Call OpenRouter API
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://yourdomain.com",
                "X-Title": "BCChatbotWebApp"
            },
            data=json.dumps({
                "model": model,
                "messages": messages
            })
        )

        if response.status_code == 200:
            reply = response.json()["choices"][0]["message"]["content"]
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("AI", reply))
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")

    # Display chat history
    for role, text in st.session_state.chat_history:
        st.markdown(f"**{role}:** {text}")

elif uploaded_file is None:
    st.info("Please upload a PDF file to begin.")
elif not api_key:
    st.warning("Enter your OpenRouter API key in the sidebar.")
