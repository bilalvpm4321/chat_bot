import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="PDF Chatbot", layout="centered")
st.title("📘 PDF Chat with AI")

uploaded_file = st.file_uploader("📂 Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    try:
        # Step 1: Extract text
        reader = PdfReader(uploaded_file)
        raw_text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                raw_text += content

        if not raw_text.strip():
            st.error("No extractable text found in the PDF.")
        else:
            # Step 2: Split text into chunks
            splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            texts = splitter.split_text(raw_text)

            # Step 3: Create embeddings and store in vector store
            embeddings = OpenAIEmbeddings(openai_api_key=openai_key)

            db = FAISS.from_texts(texts, embeddings)

            # Step 4: Chat interface
            st.subheader("💬 Ask Questions About the PDF")
            query = st.text_input("Type your question here:")

            if query:
                docs = db.similarity_search(query)
                if not docs:
                    st.warning("No relevant information found in the PDF for your question.")
                else:
                    llm = ChatOpenAI(temperature=0, openai_api_key=openai_key)
                    chain = load_qa_chain(llm, chain_type="stuff")
                    response = chain.run(input_documents=docs, question=query)

                    st.write("### 📄 Answer")
                    st.write(response)
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a PDF file to get started.")
