import streamlit as st
from pypdf import PdfReader

st.title("Enterprise RAG Assistant")
st.write("Upload a PDF and ask questions about the document.")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

def split_into_chunks(text, chunk_size=500):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks

def find_best_chunk(question, chunks):
    question_words = question.lower().split()

    best_chunk = ""
    best_score = 0

    for chunk in chunks:
        chunk_lower = chunk.lower()
        score = 0

        for word in question_words:
            if word in chunk_lower:
                score += 1

        if score > best_score:
            best_score = score
            best_chunk = chunk

    return best_chunk, best_score

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

        if score > 0:
            st.subheader("Best Matching Answer Section")
            st.write(answer_chunk)
        else:
            st.warning("I could not find relevant information in the document.")