import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if the API key is loaded correctly
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("API key not found in environment variables. Please check 'key.env'.")

# Configure the API key
genai.configure(api_key=GOOGLE_API_KEY)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Instructions:
    You are an AI expert specializing in concrete. You are provided with the"ACI-301-specs-structural-concrete.pdf", "ACI 302.1R.pdf" and "ACI_117_06_Specifications_for_Tolerances.pdf". Your task is to provide precise and actionable advice based on the user's specific questions about concrete, mixing, application, and related topics.

    Please:
    - Carefully analyze the provided context from the manual.
    - Offer tailored advice that directly addresses the user's questions.
    - Ensure that your advice is accurate and follows best practices in the concrete industry.
    - If information is missing or unclear, make logical assumptions based on the context to provide the best answer.
    - Be concise but thorough, offering detailed steps or explanations when necessary to help the user.

    Context:\n{context}\n
    Question: \n{question}\n

    Expert Advice:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    
    # Check if FAISS index exists before loading
    faiss_index_path = "faiss_index/index.faiss"
    if not os.path.exists(faiss_index_path):
        st.error("FAISS index not found. Please ensure the documents are processed first.")
        return
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    # Display the appropriate response
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="Concrete Expert Advisor")
    st.header("🌫️🧱 ACI-Concrete Expert Advisor")

    # Input field for user's question
    conc = st.text_input("What Would You Like to Know About Concrete and Its Usage?")
    
    # Show an example input below the input field
    st.caption("Example: 'How can I determine the appropriate water-cement ratio?'")

    if conc:
        user_question = f"{conc}. And Please give me guideline what will be best in this context. Give Precise instruction, Stating which ACI code it came from."
        user_input(user_question)

    with st.sidebar:
        st.title("Documents:")

        # Check if the FAISS index exists
        if not os.path.exists("faiss_index/index.faiss"):
            with st.spinner("Processing documents..."):
                # Process the provided PDFs
                pdf_files = ["ACI-301-specs-structural-concrete.pdf", "ACI 302.1R.pdf", "ACI_117_06_Specifications_for_Tolerances.pdf"]
                raw_text = get_pdf_text([open(pdf, "rb") for pdf in pdf_files])
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)  # This creates the FAISS index
            st.success("Documents processed and FAISS index created.")
        else:
            st.info("FAISS index loaded from 'faiss_index'.")

if __name__ == "__main__":
    main()
