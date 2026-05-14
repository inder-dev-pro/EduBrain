import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import time
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from prometheus_client import Counter, Histogram, Gauge, REGISTRY
from metrics_server import start_metrics_server

# Load environment variables
load_dotenv()

# ---------------------------------------------------------------------------
# Prometheus metrics definitions
# ---------------------------------------------------------------------------

def _get_or_create_metric(metric_class, name, description, *args, **kwargs):
    """Return an existing metric from the registry or create a new one.

    Streamlit re-runs the entire script on every interaction, so we guard
    against duplicate-registration errors by reusing already-registered metrics.
    """
    try:
        return metric_class(name, description, *args, **kwargs)
    except ValueError:
        # Metric already registered — retrieve it from the collector registry.
        return REGISTRY._names_to_collectors.get(name)


QUESTIONS_TOTAL = _get_or_create_metric(
    Counter,
    "edubrain_questions_total",
    "Total number of questions asked by users",
)

QUESTIONS_ERRORS_TOTAL = _get_or_create_metric(
    Counter,
    "edubrain_question_errors_total",
    "Total number of errors while answering questions",
)

PDF_PROCESSED_TOTAL = _get_or_create_metric(
    Counter,
    "edubrain_pdfs_processed_total",
    "Total number of PDF processing jobs completed successfully",
)

PDF_PROCESSING_ERRORS_TOTAL = _get_or_create_metric(
    Counter,
    "edubrain_pdf_processing_errors_total",
    "Total number of PDF processing jobs that failed",
)

QUESTION_LATENCY = _get_or_create_metric(
    Histogram,
    "edubrain_question_latency_seconds",
    "Time taken (seconds) to answer a user question",
    buckets=[0.5, 1, 2, 5, 10, 30, 60],
)

PDF_PROCESSING_LATENCY = _get_or_create_metric(
    Histogram,
    "edubrain_pdf_processing_latency_seconds",
    "Time taken (seconds) to process and index uploaded PDFs",
    buckets=[1, 5, 10, 30, 60, 120, 300],
)

FAISS_DOCS_COUNT = _get_or_create_metric(
    Gauge,
    "edubrain_faiss_documents_total",
    "Number of document chunks currently stored in the FAISS index",
)

# ---------------------------------------------------------------------------
# Start the background Prometheus metrics server (once per process)
# ---------------------------------------------------------------------------

if "metrics_server_started" not in st.session_state:
    start_metrics_server(port=9091)
    st.session_state["metrics_server_started"] = True


# ---------------------------------------------------------------------------
# Helper: API key resolution
# ---------------------------------------------------------------------------

def get_api_key():
    # Try getting from environment variable first (e.g., set via EC2 / k8s secret)
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        return api_key

    # Fallback to Streamlit secrets if available
    try:
        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core application logic
# ---------------------------------------------------------------------------

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks


def get_embeddings():
    # Local embedding model avoids requiring a second hosted API for vectorization.
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_vector_store(chunks):
    embeddings = get_embeddings()
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    doc_count = len(vector_store.index_to_docstore_id)
    FAISS_DOCS_COUNT.set(doc_count)
    st.success(f"✅ Total documents in FAISS: {doc_count}")


def get_qa_chain():
    prompt_template = """Answer the question as detailed as possible from the given context. Make sure to provide all the details and information requested in the question.
    You can give the answers from your own too according to the reference context.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, api_key=get_api_key())
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    QUESTIONS_TOTAL.inc()
    start = time.time()
    try:
        embeddings = get_embeddings()
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question, k=5)
        chain = get_qa_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.subheader("📢 AI Response:")
        st.success(response["output_text"])
    except Exception as e:
        QUESTIONS_ERRORS_TOTAL.inc()
        st.error(f"❌ Error while answering: {e}")
    finally:
        QUESTION_LATENCY.observe(time.time() - start)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="EduBrain", page_icon="📚")
    st.title("📚 Your Study Companion Using Groq")

    with st.sidebar:
        st.title("📂 Upload & Process Notes(PDF)")
        groq_api_key = get_api_key()
        if not groq_api_key:
            st.warning("⚠️ Add GROQ_API_KEY in .env or Streamlit secrets before processing.")
        pdf_docs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
        if st.button("Submit & Process"):
            if not groq_api_key:
                st.error("❌ Missing GROQ_API_KEY. Please set it and retry.")
                return
            if not pdf_docs:
                st.warning("⚠️ Please upload at least one PDF.")
                return
            start = time.time()
            try:
                with st.spinner("Processing your PDF..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ Processing Completed!")
                PDF_PROCESSED_TOTAL.inc()
            except Exception as e:
                PDF_PROCESSING_ERRORS_TOTAL.inc()
                st.error(f"❌ PDF processing failed: {e}")
            finally:
                PDF_PROCESSING_LATENCY.observe(time.time() - start)

    st.subheader("💬 Ask a Question")
    user_question = st.text_input("Type your question below:")
    if st.button("Get Answer"):
        if user_question.strip():
            if not get_api_key():
                st.error("❌ Missing GROQ_API_KEY. Please set it and retry.")
                return
            user_input(user_question)
        else:
            st.warning("⚠️ Please enter a valid question!")


if __name__ == "__main__":
    main()
