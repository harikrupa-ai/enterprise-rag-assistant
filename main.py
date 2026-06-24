import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Resume Intelligence Platform",
    page_icon="🤖",
    layout="centered"
)

st.title("AI Resume Intelligence Platform")
st.write(
    "Analyze resumes against job descriptions using semantic search, ATS keyword matching, "
    "resume improvement suggestions, and interview preparation."
)

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
        "prompt engineering", "aws", "azure", "gcp", "docker", "kubernetes",
        "fastapi", "streamlit", "api", "git", "github", "pandas", "numpy",
        "scikit-learn", "power bi", "tableau", "excel", "jira", "confluence",
        "agile", "scrum", "requirements gathering", "stakeholder management",
        "user stories", "uat", "business analysis", "data analysis", "etl",
        "data visualization"
    ]


def analyze_keywords(resume_text, job_text):
    resume_lower = resume_text.lower()
    job_lower = job_text.lower()

    required = []
    matching = []
    missing = []

    for keyword in get_keywords():
        if keyword in job_lower:
            required.append(keyword)
            if keyword in resume_lower:
                matching.append(keyword)
            else:
                missing.append(keyword)

    ats_score = round((len(matching) / len(required)) * 100, 2) if required else 0
    return ats_score, required, matching, missing


def generate_recommendation(semantic_score, ats_score):
    avg = (semantic_score + ats_score) / 2

    if avg >= 75:
        return "Strong match. This candidate is well aligned with the role."
    elif avg >= 55:
        return "Moderate match. The resume has relevant experience but needs keyword and positioning improvements."
    else:
        return "Low match. The resume needs stronger alignment with the job description."


def generate_resume_suggestions(missing_keywords):
    if not missing_keywords:
        return ["Resume is well aligned. Add measurable project outcomes to improve impact."]

    return [
        f"Add '{keyword.title()}' naturally in your skills, projects, or experience section if you have relevant experience."
        for keyword in missing_keywords[:8]
    ]


def generate_resume_bullets(matching_keywords, missing_keywords):
    bullets = []

    if "rag" in missing_keywords or "retrieval augmented generation" in missing_keywords or "rag" in matching_keywords:
        bullets.append("Built a Retrieval-Augmented Generation application using document chunking, embeddings, semantic search, and vector retrieval.")

    if "faiss" in missing_keywords or "faiss" in matching_keywords or "vector database" in missing_keywords:
        bullets.append("Implemented FAISS-based vector search to retrieve relevant document sections using sentence embeddings and similarity search.")

    if "streamlit" in missing_keywords or "streamlit" in matching_keywords:
        bullets.append("Developed an interactive Streamlit web application for resume analysis, ATS optimization, and interview preparation.")

    if "python" in missing_keywords or "python" in matching_keywords:
        bullets.append("Built Python-based workflows for PDF parsing, skill extraction, semantic matching, and candidate-job analysis.")

    if not bullets:
        bullets.append("Improved resume-job alignment by identifying skill gaps, matching keywords, and generating targeted improvement suggestions.")

    return bullets


def generate_interview_questions(matching_keywords, missing_keywords):
    technical = []

    for skill in matching_keywords[:5]:
        technical.append(f"Can you explain your experience with {skill.title()}?")

    for skill in missing_keywords[:5]:
        technical.append(f"How would you approach learning or applying {skill.title()} for this role?")

    project = [
        "Explain how your AI Resume Intelligence Platform works end to end.",
        "Why did you use Sentence Transformers?",
        "What is FAISS and why is it useful for semantic search?",
        "How does semantic search differ from keyword search?",
        "How would you scale this application for multiple users?"
    ]

    behavioral = [
        "Tell me about yourself.",
        "Why are you interested in this role?",
        "Tell me about a challenging project you worked on.",
        "Describe a time you worked with stakeholders.",
        "Why should we hire you?"
    ]

    return technical[:8], project, behavioral


if resume_file and job_description:
    with st.spinner("Analyzing resume and job description..."):
        model = load_embedding_model()
        resume_text = extract_pdf_text(resume_file)

        semantic_score = calculate_semantic_score(resume_text, job_description, model)
        ats_score, required_keywords, matching_keywords, missing_keywords = analyze_keywords(
            resume_text,
            job_description
        )

        recommendation = generate_recommendation(semantic_score, ats_score)
        suggestions = generate_resume_suggestions(missing_keywords)
        bullets = generate_resume_bullets(matching_keywords, missing_keywords)
        technical_qs, project_qs, behavioral_qs = generate_interview_questions(
            matching_keywords,
            missing_keywords
        )

    st.header("Analysis Dashboard")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Semantic Match Score", f"{semantic_score}%")
    with col2:
        st.metric("ATS Keyword Score", f"{ats_score}%")

    st.subheader("Hiring Recommendation")
    st.write(recommendation)

    st.subheader("Matching Keywords")
    if matching_keywords:
        for keyword in matching_keywords:
            st.write("✅", keyword.title())
    else:
        st.write("No matching keywords found.")

    st.subheader("Missing Keywords")
    if missing_keywords:
        for keyword in missing_keywords:
            st.write("❌", keyword.title())
    else:
        st.write("No major missing keywords found.")

    st.subheader("Resume Improvement Suggestions")
    for suggestion in suggestions:
        st.write("💡", suggestion)

    st.subheader("Suggested Resume Bullet Points")
    for bullet in bullets:
        st.write("•", bullet)

    st.subheader("Technical Interview Questions")
    for i, q in enumerate(technical_qs, start=1):
        st.write(f"{i}. {q}")

    st.subheader("Project-Based Interview Questions")
    for i, q in enumerate(project_qs, start=1):
        st.write(f"{i}. {q}")

    st.subheader("Behavioral Interview Questions")
    for i, q in enumerate(behavioral_qs, start=1):
        st.write(f"{i}. {q}")

    with st.expander("Resume Text Preview"):
        st.write(resume_text[:2000])