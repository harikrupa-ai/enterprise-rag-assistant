import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.title("AI Interview Preparation Engine")
st.write("Upload your resume and paste a job description to generate interview questions and readiness analysis.")

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


def get_skill_keywords():
    return [
        "python", "sql", "machine learning", "nlp", "llm", "generative ai",
        "rag", "langchain", "faiss", "vector database", "embeddings",
        "semantic search", "prompt engineering", "aws", "azure", "docker",
        "fastapi", "streamlit", "pandas", "numpy", "scikit-learn",
        "power bi", "tableau", "excel", "jira", "confluence", "agile",
        "scrum", "requirements gathering", "stakeholder management",
        "user stories", "uat", "business analysis", "data analysis",
        "git", "github"
    ]


def analyze_skills(resume_text, job_text):
    resume_lower = resume_text.lower()
    job_lower = job_text.lower()

    matching = []
    missing = []

    for skill in get_skill_keywords():
        if skill in job_lower:
            if skill in resume_lower:
                matching.append(skill)
            else:
                missing.append(skill)

    return matching, missing


def interview_readiness_score(match_score, matching_skills, missing_skills):
    score = match_score

    if len(missing_skills) <= 3:
        score += 5
    elif len(missing_skills) >= 8:
        score -= 10

    if len(matching_skills) >= 6:
        score += 5

    return max(0, min(100, round(score, 2)))


def generate_technical_questions(matching_skills, missing_skills):
    questions = []

    for skill in matching_skills[:6]:
        questions.append(f"Can you explain your experience with {skill.title()}?")

    for skill in missing_skills[:4]:
        questions.append(f"How would you approach learning or applying {skill.title()} for this role?")

    questions.extend([
        "Can you explain the architecture of one technical project from your resume?",
        "How would you debug a production issue in an AI or data application?",
        "How would you explain your project to a non-technical stakeholder?"
    ])

    return questions[:10]


def generate_behavioral_questions():
    return [
        "Tell me about yourself and your background.",
        "Why are you interested in this role?",
        "Tell me about a challenging project you worked on.",
        "Describe a time you had to work with stakeholders.",
        "Tell me about a time you had to learn a new tool quickly.",
        "Describe a time you handled changing requirements.",
        "Tell me about a time you worked under pressure.",
        "Why should we hire you?"
    ]


def generate_project_questions():
    return [
        "Explain how your Enterprise RAG Assistant works end to end.",
        "Why did you use Sentence Transformers in your project?",
        "What is FAISS and why did you use it?",
        "How does semantic search differ from keyword search?",
        "How does your resume screener calculate match score?",
        "What limitations does your current project have?",
        "How would you scale your project for multiple users?",
        "How would you improve this project using a production LLM API?"
    ]


def generate_focus_areas(missing_skills):
    if not missing_skills:
        return ["Review your projects, metrics, and storytelling before interviews."]

    return [f"Prepare a basic explanation of {skill.title()}." for skill in missing_skills[:6]]


if resume_file and job_description:
    with st.spinner("Preparing interview analysis..."):
        model = load_embedding_model()
        resume_text = extract_pdf_text(resume_file)

        match_score = calculate_match_score(resume_text, job_description, model)
        matching_skills, missing_skills = analyze_skills(resume_text, job_description)
        readiness = interview_readiness_score(match_score, matching_skills, missing_skills)

        technical_questions = generate_technical_questions(matching_skills, missing_skills)
        behavioral_questions = generate_behavioral_questions()
        project_questions = generate_project_questions()
        focus_areas = generate_focus_areas(missing_skills)

    st.subheader("Interview Readiness Score")
    st.metric("Readiness Score", f"{readiness}%")
    st.write(f"Resume-to-job semantic match score: {match_score}%")

    st.subheader("Strong Areas")
    if matching_skills:
        for skill in matching_skills:
            st.write("✅", skill.title())
    else:
        st.write("No strong keyword matches found.")

    st.subheader("Focus Areas Before Interview")
    for area in focus_areas:
        st.write("⚠️", area)

    st.subheader("Technical Interview Questions")
    for i, question in enumerate(technical_questions, start=1):
        st.write(f"{i}. {question}")

    st.subheader("Behavioral Interview Questions")
    for i, question in enumerate(behavioral_questions, start=1):
        st.write(f"{i}. {question}")

    st.subheader("Project-Based Questions")
    for i, question in enumerate(project_questions, start=1):
        st.write(f"{i}. {question}")

    with st.expander("Resume Text Preview"):
        st.write(resume_text[:2000])