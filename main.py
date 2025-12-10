# --- Import necessary libraries ---
import configparser
import streamlit as st
import pickle
import logging
from datetime import datetime
import os

# LangChain + ecosystem
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from langgraph.prebuilt import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from streamlit.components.v1 import html

# ------------------- Streamlit Page Settings -------------------
st.set_page_config(page_title="Smart Scheme Research App", layout="wide")

# ------------------- Global CSS -------------------
st.markdown("""
<style>
body { background-color: #0d1117; color: white; }
.block-container {
  padding-top: 2rem;
  background: linear-gradient(to bottom, #0d1117 0%, #161b22 60%, #0d1117 100%);
  min-height: 100vh;
}
[data-testid="stSidebar"] {
  background-color: #161b22; border-right: 1px solid #30363d;
  display: flex; flex-direction: column; justify-content: center; padding-top: 2rem;
}
[data-testid="stSidebar"] label { color: #c9d1d9 !important; }
[data-testid="stSidebar"] .st-radio div { padding: 0.4rem; border-radius: 6px; transition: 0.2s; }
[data-testid="stSidebar"] .st-radio div:hover { background-color: #1f6feb; color: white !important; }
[data-testid="stFileUploader"] {
  border: 2px dashed #444c56; background-color: #0d1117; color: white;
  border-radius: 10px; padding: 10px;
}
button { border-radius: 8px !important; font-weight: 600 !important; }
div.stButton > button:first-child {
  background-color: #8957e5 !important; color: white !important; border: none !important;
  padding: 0.6rem 1.5rem; width: auto !important;
}
div.stButton > button:first-child:hover { background-color: #a371f7 !important; transform: scale(1.05); }
details { background-color: #161b22 !important; border-radius: 8px; border: 1px solid #30363d !important; padding: 10px; margin-top: 10px; }
summary { color: #58a6ff; font-weight: bold; }
input, textarea {
  background-color: #0d1117 !important; color: white !important; border: 1px solid #30363d !important; border-radius: 8px !important;
}
.question-box { font-size: 1.15rem; line-height: 1.7rem; margin-bottom: 1rem; color: #c9d1d9; }
a { color: #58a6ff !important; text-decoration: underline; }
h1, h2, h3 { color: white; }
#fixed-title { text-align: center; margin-bottom: 2rem; }
#fixed-title h1 { font-size: 2.5rem; color: white; margin-bottom: 0.3rem; }
.subtitle-with-line {
  position: relative; display: inline-block; color: #8b949e; font-size: 0.95rem; padding-bottom: 10px; text-align: center;
}
.subtitle-with-line::after {
  content: ""; position: absolute; left: 50%; transform: translateX(-50%); bottom: 0;
  width: 40vw; max-width: 300px; height: 4px; background: linear-gradient(to right, #8b949e, #0d1117);
  border-radius: 20px; opacity: 0.6;
}
</style>
""", unsafe_allow_html=True)

# ------------------- Utility: Make uploaded PDF links absolute -------------------
def serve_uploaded_files():
    html("""
    <script>
    window.addEventListener("DOMContentLoaded", () => {
        const observer = new MutationObserver(() => {
            const anchors = document.querySelectorAll("a[href^='/uploads/']");
            anchors.forEach(a => {
                if (!a.href.includes(window.location.origin)) {
                    a.href = window.location.origin + a.getAttribute("href");
                }
            });
        });
        observer.observe(document.body, { childList: true, subtree: true });
    });
    </script>
    """, height=0)

# ------------------- API Key -------------------
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@st.cache_resource(show_spinner=False)
def get_cached_llm(model_name: str, api_key: str):
    return ChatGroq(model=model_name, api_key=api_key)

# ------------------- Logging -------------------
log_filename = f"logs/scheme_tool_{datetime.now().strftime('%Y-%m-%d')}.log"
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=log_filename,
    filemode="a",
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)

if "logged_once" not in st.session_state:
    logging.info("=" * 50)
    logging.info("NEW SESSION STARTED")
    logging.info("Application started.")
    if GROQ_API_KEY:
        logging.info("Groq API key loaded.")
    st.session_state.logged_once = True
    if "api_key_logged" not in st.session_state and GROQ_API_KEY:
        st.session_state.api_key_logged = True
    if "model_logged" not in st.session_state:
        model_selected = st.session_state.get("selected_model", "N/A")
        logging.info(f"Model selected by user: {model_selected}")
        st.session_state.model_logged = True

# ------------------- Guards -------------------
def check_api_key():
    if not GROQ_API_KEY:
        logging.error("Groq API key missing. Cannot proceed.")
        st.error("Missing Groq API Key. Please add it to your environment (.env).")
        st.stop()

# ------------------- Load PDF -------------------
def read_uploaded_pdf(pdf_file):
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", pdf_file.name)
    with open(file_path, "wb") as f:
        f.write(pdf_file.read())
    logging.info(f"Uploaded file saved: {file_path}")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    for page in pages:
        page.metadata["source"] = f"/uploads/{pdf_file.name}"
    logging.info(f"PDF loaded with {len(pages)} pages.")
    return pages

# ------------------- Load & Split URLs -------------------
def fetch_and_split_documents(url_list):
    try:
        with st.spinner("Fetching content from URLs..."):
            loader = UnstructuredURLLoader(urls=url_list)
            raw_docs = loader.load()
            if not raw_docs or all(not doc.page_content.strip() for doc in raw_docs):
                logging.warning("No usable content found from URLs.")
                return None
            splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
            split_docs = splitter.split_documents(raw_docs)
            logging.info(f"Split into {len(split_docs)} chunks from URLs.")
            return split_docs
    except Exception as e:
        logging.error(f"Error loading documents: {e}")
        st.error(f"Error loading content: {e}")
        return None

# ------------------- Build FAISS -------------------
def store_in_faiss(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_index = FAISS.from_documents(text_chunks, embeddings)
    vector_index.save_local("faiss_store_openai")
    logging.info("FAISS vector store created and saved using save_local().")
    return vector_index

# ------------------- Create Document QA Chain -------------------
def build_retrieval_chain(llm, vector_store):
    """
    Creates a retrieval pipeline using the new LangChain API:
    - create_stuff_documents_chain for combining context
    - create_retrieval_chain to connect retriever + LLM
    """
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Use ONLY the provided context to answer.\n\n"
        "Context:\n{context}\n\n"
        "Question: {input}\n\n"
        "Answer concisely and factually."
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    return create_retrieval_chain(retriever, document_chain)

# ------------------- Summaries for fixed questions -------------------
def summarize_sections(index):
    groq_llm = get_cached_llm(st.session_state["selected_model"], GROQ_API_KEY)
    qa_pipeline = build_retrieval_chain(groq_llm, index)

    questions = {
        "Benefits": "Summarize the key benefits of the scheme.",
        "Process": "Describe the application process for the scheme.",
        "Eligibility": "What are the eligibility criteria for this scheme?",
        "Documents": "List documents required to apply."
    }

    summary = {}
    for key, prompt in questions.items():
        result = qa_pipeline.invoke({"input": prompt})
        # New API returns keys: "answer", "context"
        text = (result.get("answer") or "").strip()
        summary[key] = text if text else "No information found."
    return summary

# ------------------- Streamlit App -------------------
def run_app():
    check_api_key()
    serve_uploaded_files()

    st.markdown("""
<div style="text-align: center; margin-top: -45px;">
  <h1>Scheme Research Application</h1>
  <span class='subtitle-with-line'>Built with LangChain, HuggingFace, FAISS and Groq</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Button styling
    st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #238636; color: white; border-radius: 8px;
        font-weight: bold; height: 3em; width: 100%; border: none; transition: 0.3s ease-in-out;
    }
    div.stButton > button:first-child:hover { background-color: #2ea043; transform: scale(1.02); }
    </style>
    """, unsafe_allow_html=True)

    # Init session state
    st.session_state.setdefault("qa_history", [])
    st.session_state.setdefault("summary_generated", False)
    st.session_state.setdefault("vector_store", None)
    st.session_state.setdefault("summary_data", None)

    # Sidebar
    with st.sidebar:
        st.header("Select Input Type")
        input_type = st.radio("Choose input method:", ("None", "URLs", "PDF"), index=0)

        if input_type == "URLs":
            st.subheader("🔗 Enter Scheme URLs")
            raw_links = st.text_area("Paste one URL per line")
            url_file = None
        elif input_type == "PDF":
            st.subheader("Upload PDF File")
            url_file = st.file_uploader("", type=["pdf"])
            raw_links = None
        else:
            raw_links = None
            url_file = None

        # Model picker
        model_choice = st.radio("Choose Model:", ["🟢 llama3-8b (Fast)", "🔵 llama3-70b (Accurate)"], index=0)
        model_map = {
            "🟢 llama3-8b (Fast)": "llama3-8b-8192",
            "🔵 llama3-70b (Accurate)": "llama3-70b-8192"
        }
        st.session_state["selected_model"] = model_map[model_choice]

        st.markdown("---")
        start_processing = st.button("Process")
        if st.button("Reset Tool"):
            # Safely clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            logging.info("User triggered Reset Tool.")
            # Optional: clean uploaded files
            upload_dir = "uploads"
            if os.path.exists(upload_dir):
                for file in os.listdir(upload_dir):
                    file_path = os.path.join(upload_dir, file)
                    try:
                        os.remove(file_path)
                        logging.info(f"Deleted uploaded file during reset: {file_path}")
                    except Exception as e:
                        logging.warning(f"Could not delete {file_path}: {e}")
            st.rerun()

    # Process inputs
    if start_processing:
        all_documents = []
        logging.info("Processing started.")

        if raw_links:
            urls = [line.strip() for line in raw_links.splitlines() if line.strip()]
            if urls:
                url_docs = fetch_and_split_documents(urls)
                if url_docs:
                    all_documents.extend(url_docs)

        if url_file:
            logging.info(f"PDF file uploaded: {url_file.name}")
            pdf_docs = read_uploaded_pdf(url_file)
            if pdf_docs:
                splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
                split_pdf = splitter.split_documents(pdf_docs)
                logging.info(f"PDF split into {len(split_pdf)} chunks.")
                all_documents.extend(split_pdf)

        logging.info(f"Total documents collected for embedding: {len(all_documents)}")

        if all_documents:
            with st.spinner("⚡ Powering up your scheme assistant..."):
                vector_db = store_in_faiss(all_documents)
                st.session_state.vector_store = vector_db
        else:
            st.warning("⚠ Could not extract usable text from the given inputs.")
            logging.warning("No documents found to process.")

    # Generate summary
    if st.session_state.vector_store:
        st.subheader("Comprehensive Scheme Summary")
        if st.button("Generate Summary"):
            logging.info("User clicked Generate Summary.")
            with st.spinner("Summarizing scheme details..."):
                logging.info("Summary generation started.")
                summary_data = summarize_sections(st.session_state.vector_store)
                st.session_state.summary_data = summary_data
                st.session_state.summary_generated = True
                logging.info("Summary generation completed.")

    if st.session_state.summary_generated and st.session_state.summary_data:
        st.subheader("Scheme Summary")
        st.caption(f"Generated using: `{st.session_state.get('selected_model', 'llama3-8b-8192')}`")

        with st.expander("📌 Scheme Benefits", expanded=True):
            st.write(st.session_state.summary_data.get("Benefits", "No information found."))

        with st.expander("📌 Application Process", expanded=True):
            st.write(st.session_state.summary_data.get("Process", "No information found."))

        with st.expander("📌 Eligibility Criteria", expanded=True):
            st.write(st.session_state.summary_data.get("Eligibility", "No information found."))

        with st.expander("📌 Required Documents", expanded=True):
            st.write(st.session_state.summary_data.get("Documents", "No information found."))

    # Free-form Q&A
    if st.session_state.vector_store:
        st.subheader("Ask a Question")
        question = st.text_input("What would you like to know about the scheme?")
        if question:
            if not any(q["question"] == question for q in st.session_state.qa_history):
                with st.spinner("Processing your question..."):
                    logging.info(f"User asked: {question}")
                    logging.info(f"Answering question using model: {st.session_state['selected_model']}")

                    try:
                        llm = get_cached_llm(st.session_state["selected_model"], GROQ_API_KEY)
                        qa_chain = build_retrieval_chain(llm, st.session_state.vector_store)

                        result = qa_chain.invoke({"input": question})
                        answer = (result.get("answer") or "").strip()

                        if len(answer) < 10:
                            logging.warning(f"Short or empty answer returned for: {question}")

                        # New API returns "context" (list[Document]) instead of "source_documents"
                        source_urls = set()
                        for doc in result.get("context", []):
                            metadata = getattr(doc, "metadata", {})
                            if "source" in metadata:
                                source_urls.add(metadata["source"])

                        st.session_state.qa_history.append({
                            "question": question,
                            "answer": answer if answer else "No information found.",
                            "sources": list(source_urls)
                        })

                        st.session_state.current_question = ""
                        logging.info(f"Appended QA to session state. Total: {len(st.session_state.qa_history)}")
                        st.rerun()

                    except Exception as e:
                        logging.error(f"Error while answering question: {e}")
                        st.error(f"Error answering the question. Please try again.\n{e}")

        for idx, pair in enumerate(st.session_state.qa_history):
            with st.expander(f"Q{idx + 1}: {pair['question']}"):
                st.markdown(f"""
    <div class='question-box'>
        <strong>Answer:</strong> {pair['answer']}<br>
        <span style="font-size: 0.8em; color: gray;">Model: {st.session_state.get("selected_model", "llama3-8b-8192")}</span>
    </div>
""", unsafe_allow_html=True)

                if pair.get("sources"):
                    urls = [src for src in pair["sources"] if not src.endswith(".pdf")]
                    pdfs = [src for src in pair["sources"] if src.endswith(".pdf")]

                    # Inline web URLs
                    if urls:
                        joined_urls = " ".join(f"<a href='{u}' target='_blank'>{u}</a>" for u in urls)
                        st.markdown(f"<p>🔗 <b>Source URLs:</b> {joined_urls}</p>", unsafe_allow_html=True)

                    # Download buttons for PDFs saved locally
                    for src in pdfs:
                        file_path = "." + src
                        file_name = os.path.basename(file_path)
                        try:
                            with open(file_path, "rb") as f:
                                pdf_bytes = f.read()
                            st.markdown(
                                "<p style='font-size:1.15rem;font-weight:600;margin-bottom:0.2rem;color:#c9d1d9;'> 🔗 Source PDFs:</p>",
                                unsafe_allow_html=True
                            )
                            st.download_button(
                                label=f"Download {file_name}",
                                data=pdf_bytes,
                                file_name=file_name,
                                mime="application/pdf",
                                key=f"download_{idx}_{file_name}"
                            )
                            logging.info(f"Displayed download button for {file_name} in Q{idx + 1}.")
                        except FileNotFoundError:
                            logging.warning(f"PDF file not found: {file_name}")
                            st.warning(f"⚠ File not found: {file_name}")

# ------------------- Entrypoint -------------------
if __name__ == "__main__":
    run_app()

