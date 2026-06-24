import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.title("ATS Keyword Optimizer")
st.write("Upload a resume and paste a job description to get ATS score, keyword gaps, and optimization suggestions.")

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


def calculate_semantic_score(resume_text, job_text, model):
    resume_embedding = model.encode([resume_text])
    job_embedding = model.encode([job_text])

    score = cosine_similarity(resume_embedding, job_embedding)[0][0]
    return round(score * 100, 2)


def get_keywords():
    return [
        "python", "sql", "machine learning", "deep learning", "nlp", "llm",
        "generative ai", "rag", "retrieval augmented generation", "langchain",
        "faiss", "chromadb", "vector database", "embeddings", "semantic search",
        "prompt engineering", "openai", "gemini", "aws", "azure", "gcp",
        "docker", "kubernetes", "fastapi", "streamlit", "api", "git", "github",
        "pandas", "numpy", "scikit-learn", "power bi", "tableau", "excel",
        "jira", "confluence", "agile", "scrum", "requirements gathering",
        "stakeholder management", "user stories", "uat", "business analysis",
        "data analysis", "etl", "data visualization"
    ]


def analyze_keywords(resume_text, job_text):
    resume_lower = resume_text.lower()
    job_lower = job_text.lower()

    required_keywords = []
    matching_keywords = []
    missing_keywords = []

    for keyword in get_keywords():
        if keyword in job_lower:
            required_keywords.append(keyword)

            if keyword in resume_lower:
                matching_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)

    if required_keywords:
        ats_score = round((len(matching_keywords) / len(required_keywords)) * 100, 2)
    else:
        ats_score = 0

    return ats_score, required_keywords, matching_keywords, missing_keywords


def priority_level(keyword):
    high_priority = [
        "python", "machine learning", "nlp", "llm", "generative ai",
        "rag", "langchain", "faiss", "vector database", "embeddings",
        "semantic search", "prompt engineering", "aws"
    ]

    if keyword in high_priority:
        return "High"
    return "Medium"


def generate_keyword_suggestions(missing_keywords):
    suggestions = []

    for keyword in missing_keywords:
        suggestions.append(
            f"Add '{keyword.title()}' naturally in your skills, projects, or experience section if you have relevant exposure."
        )

    return suggestions


def generate_optimized_bullets(missing_keywords):
    bullets = []

    if "rag" in missing_keywords or "retrieval augmented generation" in missing_keywords:
        bullets.append("Built a Retrieval-Augmented Generation application using PDF processing, semantic search, and vector-based document retrieval.")

    if "faiss" in missing_keywords or "vector database" in missing_keywords:
        bullets.append("Implemented FAISS vector search to retrieve relevant document chunks using sentence embeddings and similarity search.")

    if "embeddings" in missing_keywords or "semantic search" in missing_keywords:
        bullets.append("Generated text embeddings using Sentence Transformers to enable semantic search across uploaded documents.")

    if "streamlit" in missing_keywords:
        bullets.append("Developed an interactive Streamlit web application for AI-powered resume analysis and document question answering.")

    if "python" in missing_keywords:
        bullets.append("Built Python-based automation workflows for document parsing, skill extraction, and resume-job matching.")

    if not bullets:
        bullets.append("Resume already contains many required keywords. Improve bullet impact by adding measurable outcomes and project results.")

    return bullets


if resume_file and job_description:
    with st.spinner("Running ATS keyword analysis..."):
        model = load_embedding_model()
        resume_text = extract_pdf_text(resume_file)

        semantic_score = calculate_semantic_score(resume_text, job_description, model)
        ats_score, required_keywords, matching_keywords, missing_keywords = analyze_keywords(
            resume_text,
            job_description
        )

        suggestions = generate_keyword_suggestions(missing_keywords)
        optimized_bullets = generate_optimized_bullets(missing_keywords)

    st.subheader("Overall Match Scores")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Semantic Match Score", f"{semantic_score}%")

    with col2:
        st.metric("ATS Keyword Score", f"{ats_score}%")

    st.subheader("Required Keywords Found in Job Description")
    if required_keywords:
        for keyword in required_keywords:
            st.write("🔎", keyword.title())
    else:
        st.write("No tracked keywords found in job description.")

    st.subheader("Matching Keywords")
    if matching_keywords:
        for keyword in matching_keywords:
            st.write("✅", keyword.title())
    else:
        st.write("No matching keywords found.")

    st.subheader("Missing Keywords")
    if missing_keywords:
        for keyword in missing_keywords:
            st.write(f"❌ {keyword.title()} — Priority: {priority_level(keyword)}")
    else:
        st.write("No major missing keywords found.")

    st.subheader("Keyword Optimization Suggestions")
    for suggestion in suggestions:
        st.write("💡", suggestion)

    st.subheader("Suggested Resume Bullet Points")
    for bullet in optimized_bullets:
        st.write("•", bullet)

    with st.expander("Resume Text Preview"):
        st.write(resume_text[:2000])