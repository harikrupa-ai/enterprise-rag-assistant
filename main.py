# PHASE 2 TEST
import streamlit as st
from pypdf import PdfReader

st.title("Enterprise RAG Assistant")
st.write("Upload a PDF and ask questions about the document.")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)

    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    st.success("PDF uploaded and text extracted successfully!")

    st.subheader("Preview of PDF Content")
    st.text_area("Extracted Text", text[:1500], height=250)

    question = st.text_input("Ask a question or enter a keyword:")

    if question:
        matches = []

        for line in text.split("\n"):
            if question.lower() in line.lower():
                matches.append(line)

        if matches:
            st.success("Found matching information!")

            for match in matches[:10]:
                st.write(match)

        else:
            st.warning("No matching information found.")
