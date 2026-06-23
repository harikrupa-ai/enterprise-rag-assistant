import os
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

st.title("Enterprise RAG Assistant")
st.write("Upload a PDF and ask AI-powered questions about the document.")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")


def split_into_chunks(text, chunk_size=500):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))

    return chunks


def find_best_chunk(question, chunks):
    question_words = question.lower().replace("?", "").split()

    related_words = {
        "skills": ["skills", "tools", "technologies", "sql", "power", "python", "jira", "excel", "confluence"],
        "experience": ["experience", "professional", "worked", "role", "company", "accenture", "business"],
        "projects": ["projects", "dashboard", "platform", "process", "enhancement", "optimization"],
        "education": ["education", "university", "master", "bachelor", "degree"],
        "tools": ["tools", "sql", "power", "bi", "excel", "jira", "confluence", "python"],
    }

    expanded_words = question_words.copy()

    for word in question_words:
        if word in related_words:
            expanded_words.extend(related_words[word])

    best_chunk = ""
    best_score = 0

    for chunk in chunks:
        chunk_lower = chunk.lower()
        score = 0

        for word in expanded_words:
            if word in chunk_lower:
                score += 1

        if score > best_score:
            best_score = score
            best_chunk = chunk

    return best_chunk, best_score


def generate_answer(question, context):
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = f"""
You are an AI assistant. Answer the user's question using only the document context below.

Document Context:
{context}

User Question:
{question}

Give a clear, short, professional answer. If the document does not contain the answer, say that clearly.
"""

    response = model.generate_content(prompt)
    return response.text


if not api_key:
    st.error("Google API key not found. Please add GOOGLE_API_KEY to your .env file.")

if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)

    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    st.success("PDF uploaded and text extracted successfully!")

    st.subheader("Preview of PDF Content")
    st.text_area("Extracted Text", text[:1500], height=250)

    chunks = split_into_chunks(text)

    question = st.text_input("Ask a question about the document:")

    if question:
        answer_chunk, score = find_best_chunk(question, chunks)

        if answer_chunk:
            st.subheader("AI Answer")

            try:
                with st.spinner("Gemini is thinking..."):
                    answer = generate_answer(question, answer_chunk)

                st.write(answer)

                st.subheader("Source Section Used")
                st.write(answer_chunk)

            except Exception as e:
                st.error("Gemini API error. Your key may be out of quota or rate-limited.")
                st.write(e)

                st.subheader("Best Matching Section Without AI")
                st.write(answer_chunk)
        else:
            st.warning("I could not find relevant information in the document.")