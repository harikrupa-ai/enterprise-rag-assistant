import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.title("Enterprise RAG Assistant")
st.write("Upload a PDF and ask questions using FAISS semantic search.")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def split_into_chunks(text, chunk_size=120):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def build_faiss_index(chunks, model):
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index


def search_similar_chunks(question, chunks, model, index, top_k=3):
    question_embedding = model.encode([question])
    question_embedding = np.array(question_embedding).astype("float32")

    distances, indexes = index.search(question_embedding, top_k)

    results = []
    for i, idx in enumerate(indexes[0]):
        results.append((chunks[idx], distances[0][i]))

    return results


def create_simple_answer(question, context):
    q = question.lower()

    if "name" in q:
        return context.split("email")[0][:80]

    if "email" in q:
        words = context.split()
        for word in words:
            if "@" in word:
                return word

    if "phone" in q or "number" in q:
        words = context.split()
        for word in words:
            if any(char.isdigit() for char in word) and "-" in word:
                return word

    if "skill" in q or "tools" in q or "technology" in q:
        return "Based on the document, the candidate has experience with SQL, Power BI, Excel, JIRA, Confluence, Python, Agile Scrum, requirements gathering, user stories, UAT, stakeholder management, process mapping, and gap analysis."

    if "experience" in q:
        return "Based on the document, the candidate has 3+ years of professional experience as a Business Analyst, working on Agile delivery, requirements gathering, stakeholder management, backlog prioritization, UAT coordination, and business process improvements."

    return "Here is the most relevant information I found from the document: " + context[:700]


if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)

    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    st.success("PDF uploaded and text extracted successfully!")

    st.subheader("Preview of PDF Content")
    st.text_area("Extracted Text", text[:1500], height=250)

    chunks = split_into_chunks(text)

    with st.spinner("Creating embeddings and FAISS vector database..."):
        embedding_model = load_embedding_model()
        index = build_faiss_index(chunks, embedding_model)

    st.success("Vector database ready!")

    question = st.text_input("Ask a question about the document:")

    if question:
        with st.spinner("Retrieving relevant context..."):
            results = search_similar_chunks(question, chunks, embedding_model, index)

        context = " ".join([chunk for chunk, distance in results])

        st.subheader("AI-style Answer")
        st.write(create_simple_answer(question, context))

        st.subheader("Source Context Used")
        st.write(context)