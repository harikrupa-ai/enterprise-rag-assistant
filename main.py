import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.title("Enterprise RAG Assistant")
st.write("Upload a PDF and ask questions using semantic vector search.")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def split_into_chunks(text, chunk_size=120):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))

    return chunks


def build_faiss_index(chunks, model):
    embeddings = model.encode(chunks)

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, embeddings


def search_similar_chunks(question, chunks, model, index, top_k=3):
    question_embedding = model.encode([question])
    question_embedding = np.array(question_embedding).astype("float32")

    distances, indexes = index.search(question_embedding, top_k)

    results = []

    for i, idx in enumerate(indexes[0]):
        results.append((chunks[idx], distances[0][i]))

    return results


if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)

    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    st.success("PDF uploaded and text extracted successfully!")

    st.subheader("Preview of PDF Content")
    st.text_area("Extracted Text", text[:1500], height=250)

    chunks = split_into_chunks(text)

    with st.spinner("Creating semantic embeddings..."):
        embedding_model = load_embedding_model()
        index, embeddings = build_faiss_index(chunks, embedding_model)

    st.success("Vector database created successfully!")

    question = st.text_input("Ask a question about the document:")

    if question:
        results = search_similar_chunks(question, chunks, embedding_model, index)

        st.subheader("Top Semantic Matches")

        for chunk, distance in results:
            st.write(f"Distance Score: {distance:.2f}")
            st.write(chunk)
            st.divider()