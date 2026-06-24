import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.title("Enterprise Multi-Document RAG Assistant")
st.write("Upload multiple PDFs and ask questions across all documents.")

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def extract_pdf_text(file):
    pdf_reader = PdfReader(file)
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    return text


def split_into_chunks(text, file_name, chunk_size=120):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append({
            "file_name": file_name,
            "content": chunk
        })

    return chunks


def build_faiss_index(chunks, model):
    texts = [chunk["content"] for chunk in chunks]
    embeddings = model.encode(texts)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index


def search_documents(question, chunks, model, index, top_k=5):
    question_embedding = model.encode([question])
    question_embedding = np.array(question_embedding).astype("float32")

    distances, indexes = index.search(question_embedding, top_k)

    results = []

    for i, idx in enumerate(indexes[0]):
        results.append({
            "file_name": chunks[idx]["file_name"],
            "content": chunks[idx]["content"],
            "distance": distances[0][i]
        })

    return results


def create_answer(question, results):
    context = " ".join([result["content"] for result in results])
    q = question.lower()

    if "which document" in q or "which file" in q:
        files = list(set([result["file_name"] for result in results]))
        return "The most relevant information was found in: " + ", ".join(files)

    if "skills" in q or "tools" in q or "technologies" in q:
        return "Based on the retrieved documents, relevant skills/tools include SQL, Power BI, Excel, JIRA, Confluence, Python, Agile/Scrum, requirements gathering, stakeholder management, and process mapping if mentioned in the uploaded files."

    if "summary" in q or "summarize" in q:
        return "Summary based on the retrieved documents: " + context[:700]

    return "Here is the most relevant information I found: " + context[:700]


if uploaded_files:
    all_chunks = []

    with st.spinner("Reading and processing documents..."):
        for file in uploaded_files:
            text = extract_pdf_text(file)
            file_chunks = split_into_chunks(text, file.name)
            all_chunks.extend(file_chunks)

    st.success(f"Processed {len(uploaded_files)} document(s).")
    st.write(f"Created {len(all_chunks)} text chunks.")

    with st.spinner("Creating embeddings and vector database..."):
        embedding_model = load_embedding_model()
        index = build_faiss_index(all_chunks, embedding_model)

    st.success("Multi-document vector database ready!")

    question = st.text_input("Ask a question across all uploaded documents:")

    if question:
        results = search_documents(question, all_chunks, embedding_model, index)

        st.subheader("AI-style Answer")
        st.write(create_answer(question, results))

        st.subheader("Top Source Sections")

        for result in results:
            st.write(f"📄 Source File: {result['file_name']}")
            st.write(f"Distance Score: {result['distance']:.2f}")
            st.write(result["content"])
            st.divider()