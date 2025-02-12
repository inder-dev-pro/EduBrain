import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) 



def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")
    st.write(f"Total documents in FAISS: {len(vector_store.index_to_docstore_id)}")

def get_qa_chain():
    prompt_template ="""Answer the question as detailed as possible from the given context, make sure to provide all the details and information that is asked in the question.,
    if the answer is not available in the context, please mention that the answer is not available in the context. Don't provide any false information.

    Context:
    {context}

    Question:
    {question}

    Answer:

    """
    model=ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain=load_qa_chain(llm=model,chain_type="stuff",prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db=FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs=new_db.similarity_search(user_question)
    chain=get_qa_chain()

    response=chain(
        {"input_documents":docs, "question":user_question},
        return_only_outputs=True)

    print(response)
    st.write("Reply:",response["output_text"])

def main():
    st.set_page_config(page_title="Question Answering System", page_icon="📚")
    st.header("Chatwith PDF using Geminni Pro")

    user_question=st.text_input("Enter your question here:")


    if user_question:
        user_input(user_question)
    with st.sidebar:
        st.title("Menu")
        pdf_docs=st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=True)
        if st.button("Submit and Process"):
            with st.spinner("Processing..."):
                raw_text=get_pdf_text(pdf_docs)
                text_chunks=get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing Completed")

if __name__=="__main__":
    main()