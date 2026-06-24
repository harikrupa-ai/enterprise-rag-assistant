import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.title("AI Resume Improvement Engine")
st.write("Upload a resume and paste a job description to get match score, missing skills, and resume improvement suggestions.")

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


def extract_skills(resume_text, job_text):
    skills = [
        "python", "sql", "power bi", "tableau", "excel", "machine learning",
        "deep learning", "nlp", "rag", "llm", "langchain", "faiss", "chromadb",
        "vector database", "embeddings", "semantic search", "aws", "azure", "gcp",
        "docker", "kubernetes", "fastapi", "flask", "streamlit", "pandas", "numpy",
        "scikit-learn", "tensorflow", "pytorch", "jira", "confluence", "agile",
        "scrum", "requirements gathering", "stakeholder management", "user stories",
        "uat", "business analysis", "data analysis", "etl", "data visualization",
        "api", "git", "github", "prompt engineering", "generative ai"
    ]

    resume_lower = resume_text.lower()
    job_lower = job_text.lower()

    matching = []
    missing = []

    for skill in skills:
        if skill in job_lower:
            if skill in resume_lower:
                matching.append(skill)
            else:
                missing.append(skill)

    return matching, missing


def generate_recommendation(score):
    if score >= 75:
        return "Strong match. This resume is well aligned with the job description."
    elif score >= 55:
        return "Moderate match. The resume has relevant experience but should be improved for better alignment."
    else:
        return "Low match. The resume needs stronger alignment with the job description."


def generate_improvement_suggestions(missing_skills):
    suggestions = []

    for skill in missing_skills[:8]:
        suggestions.append(f"Add relevant experience or project details related to {skill.title()} if you have it.")

    if not suggestions:
        suggestions.append("Resume is well aligned with the listed skills. Improve impact by adding measurable results.")

    return suggestions


def generate_resume_bullets(matching_skills, missing_skills):
    bullets = []

    if "python" in matching_skills or "python" in missing_skills:
        bullets.append("Built Python-based applications for document processing, semantic search, and automated analysis.")

    if "rag" in matching_skills or "rag" in missing_skills:
        bullets.append("Developed a Retrieval-Augmented Generation system using document chunking, embeddings, and vector search.")

    if "faiss" in matching_skills or "faiss" in missing_skills or "vector database" in matching_skills or "vector database" in missing_skills:
        bullets.append("Implemented FAISS-based vector search to retrieve relevant document sections using semantic similarity.")

    if "streamlit" in matching_skills or "streamlit" in missing_skills:
        bullets.append("Designed and deployed an interactive Streamlit web application for AI-powered resume and document analysis.")

    if "sql" in matching_skills:
        bullets.append("Used SQL to support data analysis, reporting, and business decision-making workflows.")

    if "power bi" in matching_skills:
        bullets.append("Built Power BI dashboards to track KPIs, business metrics, and operational performance.")

    if not bullets:
        bullets.append("Improved business workflows by analyzing requirements, identifying gaps, and delivering data-driven solutions.")

    return bullets


if resume_file and job_description:
    with st.spinner("Analyzing resume and job description..."):
        model = load_embedding_model()
        resume_text = extract_pdf_text(resume_file)

        score = calculate_match_score(resume_text, job_description, model)
        matching_skills, missing_skills = extract_skills(resume_text, job_description)
        recommendation = generate_recommendation(score)
        suggestions = generate_improvement_suggestions(missing_skills)
        resume_bullets = generate_resume_bullets(matching_skills, missing_skills)

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

    st.subheader("Resume Improvement Suggestions")
    for suggestion in suggestions:
        st.write("💡", suggestion)

    st.subheader("Suggested Resume Bullet Points")
    for bullet in resume_bullets:
        st.write("•", bullet)

    with st.expander("Resume Text Preview"):
        st.write(resume_text[:2000])