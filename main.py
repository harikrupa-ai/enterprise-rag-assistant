import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.title("AI Resume Screener")
st.write("Upload a resume and paste a job description to calculate semantic match score.")

resume_file = st.file_uploader("Upload Resume PDF", type="pdf")
job_description = st.text_area("Paste Job Description", height=250)


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def extract_pdf_text(file):
    pdf_reader = PdfReader(file)
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    return text


def calculate_match_score(resume_text, job_text, model):
    resume_embedding = model.encode([resume_text])
    job_embedding = model.encode([job_text])

    score = cosine_similarity(resume_embedding, job_embedding)[0][0]
    return round(score * 100, 2)


def extract_matching_keywords(resume_text, job_text):
    common_skills = [
        "python", "sql", "power bi", "tableau", "excel", "machine learning",
        "deep learning", "nlp", "rag", "langchain", "faiss", "chromadb",
        "aws", "azure", "gcp", "docker", "fastapi", "streamlit",
        "pandas", "numpy", "scikit-learn", "jira", "confluence",
        "agile", "scrum", "requirements gathering", "stakeholder management",
        "user stories", "uat", "business analysis", "data analysis",
        "etl", "data visualization", "api", "git", "github"
    ]

    resume_lower = resume_text.lower()
    job_lower = job_text.lower()

    matching = []
    missing = []

    for skill in common_skills:
        if skill in job_lower:
            if skill in resume_lower:
                matching.append(skill)
            else:
                missing.append(skill)

    return matching, missing


def generate_recommendation(score):
    if score >= 75:
        return "Strong match. Recommend interview."
    elif score >= 55:
        return "Moderate match. Candidate may be suitable with some skill gaps."
    else:
        return "Low match. Significant skill gaps found."


if resume_file and job_description:
    with st.spinner("Analyzing resume and job description..."):
        model = load_embedding_model()
        resume_text = extract_pdf_text(resume_file)

        score = calculate_match_score(resume_text, job_description, model)
        matching_skills, missing_skills = extract_matching_keywords(resume_text, job_description)
        recommendation = generate_recommendation(score)

    st.subheader("Resume Match Score")
    st.metric("Match Score", f"{score}%")

    st.subheader("Matching Skills")
    if matching_skills:
        for skill in matching_skills:
            st.write("✅", skill.title())
    else:
        st.write("No matching skills found.")

    st.subheader("Missing Skills")
    if missing_skills:
        for skill in missing_skills:
            st.write("❌", skill.title())
    else:
        st.write("No major missing skills found.")

    st.subheader("Hiring Recommendation")
    st.write(recommendation)

    with st.expander("Resume Text Preview"):
        st.write(resume_text[:2000])