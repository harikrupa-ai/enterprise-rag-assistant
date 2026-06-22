import streamlit as st
from pypdf import PdfReader

# Page Title
st.title("Enterprise RAG Assistant")
st.write("Upload a PDF and ask questions about it.")

# Upload PDF
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:

    # Read PDF
    pdf_reader = PdfReader(uploaded_file)

    # Extract text
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    # Success message
    st.success("PDF uploaded and text extracted!")

    # Show PDF content preview
    st.subheader("Preview of PDF Content")
    st.text(text[:1000])

    # Ask question
    question = st.text_input("Ask a question:")

    if question:

        # Simple keyword search
        if question.lower() in text.lower():
            st.success("I found that information in the document.")
        else:
            st.warning("I could not find that information in the document.")

