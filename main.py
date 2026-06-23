import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Enterprise RAG Assistant")
st.write("Upload a PDF and ask questions using free vector search.")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")


def split_into_chunks(text, chunk_size=120):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def find_best_chunks(question, chunks, top_k=3):
    documents = chunks + [question]

    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(documents)

    question_vector = vectors[-1]
    chunk_vectors = vectors[:-1]

    similarities = cosine_similarity(question_vector, chunk_vectors).flatten()

    ranked_indexes = similarities.argsort()[::-1]

    results = []

    for index in ranked_indexes[:top_k]:
        if similarities[index] > 0:
            results.append((chunks[index], similarities[index]))

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

    question = st.text_input("Ask a question about the document:")

    if question:
        results = find_best_chunks(question, chunks)

        if results:
            st.subheader("Best Matching Sections")

            for chunk, score in results:
                st.write(f"Similarity Score: {score:.2f}")
                st.write(chunk)
                st.divider()
        else:
            st.warning("I could not find relevant information in the document.")