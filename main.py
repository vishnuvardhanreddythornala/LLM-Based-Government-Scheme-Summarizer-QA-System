# --- Import necessary libraries ---
import configparser
import streamlit as st
import pickle
import logging
from datetime import datetime
import os

# LangChain and HuggingFace components for document loading and question answering
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# NEW: Replace RetrievalQA and chain imports
from langchain_core.prompts import ChatPromptTemplate

from streamlit.components.v1 import html

# Set page config for wide layout
st.set_page_config(page_title="Smart Scheme Research App", layout="wide")

#Css block
st.markdown("""
<style>
/* Layout Background */
body {
    background-color: #0d1117;
    color: white;
}
.block-container {
    padding-top: 2rem;
    background: linear-gradient(to bottom, #0d1117 0%, #161b22 60%, #0d1117 100%);
    min-height: 100vh;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding-top: 2rem;
}

[data-testid="stSidebar"] label {
    color: #c9d1d9 !important;
}
[data-testid="stSidebar"] .st-radio div {
    padding: 0.4rem;
    border-radius: 6px;
    transition: 0.2s ease-in-out;
}
[data-testid="stSidebar"] .st-radio div:hover {
    background-color: #1f6feb;
    color: white !important;
}

/* File Uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed #444c56;
    background-color: #0d1117;
    color: white;
    border-radius: 10px;
    padding: 10px;
}

/* Buttons */
button {
    border-radius: 8px !important;
    font-weight: 600 !important;
}
div.stButton > button:first-child {
    background-color: #8957e5 !important;
    color: white !important;
    border: none !important;
    padding: 0.6rem 1.5rem;
    width: auto !important;
}
div.stButton > button:first-child:hover {
    background-color: #a371f7 !important;
    transform: scale(1.05);
}

/* Expanders */
details {
    background-color: #161b22 !important;
    border-radius: 8px;
    border: 1px solid #30363d !important;
    padding: 10px;
    margin-top: 10px;
}
summary {
    color: #58a6ff;
    font-weight: bold;
}

/* Inputs */
input, textarea {
    background-color: #0d1117 !important;
    color: white !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}

/* QA Section */
.question-box {
    font-size: 1.15rem;
    line-height: 1.7rem;
    margin-bottom: 1rem;
    color: #c9d1d9;
}

/* Links */
a {
    color: #58a6ff !important;
    text-decoration: underline;
}

/* Headings */
h1, h2, h3 {
    color: white;
}
#fixed-title {
    text-align: center;
    margin-bottom: 2rem;
}

#fixed-title h1 {
    font-size: 2.5rem;
    color: white;
    margin-bottom: 0.3rem;
}

.subtitle-with-line {
    position: relative;
    display: inline-block;
    color: #8b949e;
    font-size: 0.95rem;
    padding-bottom: 10px;
    text-align: center;
}

.subtitle-with-line::after {
    content: "";
    position: absolute;
    left: 50%;
    transform: translateX(-50%);
    bottom: 0;
    width: 40vw;
    max-width: 300px;
    height: 4px;
    background: linear-gradient(to right, #8b949e, #0d1117);
    border-radius: 20px;
    opacity: 0.6;
}
</style>
""", unsafe_allow_html=True)

# Inject JavaScript to make uploaded PDF URLs clickable and downloadable
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

# --- Load API key ---
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@st.cache_resource(show_spinner=False)
def get_cached_llm(model_name: str, api_key: str):
    return ChatGroq(model=model_name, api_key=api_key)

# --- Logging setup ---
log_filename = f"logs/scheme_tool_{datetime.now().strftime('%Y-%m-%d')}.log"
os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename=log_filename, filemode="a",
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

# --- API Key Check ---
def check_api_key():
    if not GROQ_API_KEY:
        st.error("Missing Groq API Key. Add it to your .env file.")
        st.stop()

# --- PDF Loader ---
def read_uploaded_pdf(pdf_file):
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", pdf_file.name)
    with open(file_path, "wb") as f:
        f.write(pdf_file.read())

    loader = PyPDFLoader(file_path)
    pages = loader.load()
    for page in pages:
        page.metadata["source"] = f"/uploads/{pdf_file.name}"
    return pages

# --- URL Loader ---
def fetch_and_split_documents(url_list):
    try:
        loader = UnstructuredURLLoader(urls=url_list)
        raw_docs = loader.load()
        if not raw_docs:
            return None
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
        return splitter.split_documents(raw_docs)
    except Exception as e:
        st.error(f"Error loading content: {e}")
        return None

# --- FAISS Store ---
def store_in_faiss(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_index = FAISS.from_documents(text_chunks, embeddings)
    vector_index.save_local("faiss_store_openai")
    return vector_index

# ------------------------------------------------
# ⭐ NEW RETRIEVAL FUNCTION (REPLACES RetrievalQA)
# ------------------------------------------------
def run_retrieval_query(llm, vector_store, question):
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    docs = retriever.invoke(question)

    context = "\n\n".join(d.page_content for d in docs)

    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Use ONLY the context below.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer clearly and concisely:"
    )

    final_prompt = prompt.format(context=context, question=question)
    answer = llm.invoke(final_prompt).content

    return answer, docs

# --- Summary generation ---
def summarize_sections(index):
    llm = get_cached_llm(st.session_state["selected_model"], GROQ_API_KEY)

    questions = {
        "Benefits": "Summarize the key benefits of the scheme.",
        "Process": "Describe the application process for the scheme.",
        "Eligibility": "What are the eligibility criteria?",
        "Documents": "List the documents required."
    }

    summary = {}
    for key, prompt in questions.items():
        answer, _ = run_retrieval_query(llm, index, prompt)
        summary[key] = answer.strip() or "No information found."

    return summary

# --- Main Streamlit app ---
def run_app():
    check_api_key()
    serve_uploaded_files()

    st.markdown("""
    <div style="text-align:center;margin-top:-45px;">
        <h1>Scheme Research Application</h1>
        <span class='subtitle-with-line'>Built with LangChain, HuggingFace, FAISS and Groq</span>
    </div>
    """, unsafe_allow_html=True)

    st.session_state.setdefault("qa_history", [])
    st.session_state.setdefault("summary_generated", False)
    st.session_state.setdefault("vector_store", None)
    st.session_state.setdefault("summary_data", None)

    # Sidebar Inputs
    with st.sidebar:
        st.header("Select Input Type")
        input_type = st.radio("Choose input:", ("None", "URLs", "PDF"))

        raw_links = None
        url_file = None

        if input_type == "URLs":
            raw_links = st.text_area("Paste URLs (one per line)")
        elif input_type == "PDF":
            url_file = st.file_uploader("Upload PDF", type=["pdf"])

        # Model Selector
        model_choice = st.radio("Choose Model:",
                                ["🟢 llama3-8b (Fast)", "🔵 llama3-70b (Accurate)"])
        model_map = {
            "🟢 llama3-8b (Fast)": "llama3-8b-8192",
            "🔵 llama3-70b (Accurate)": "llama3-70b-8192"
        }
        st.session_state["selected_model"] = model_map[model_choice]

        start_processing = st.button("Process")

        if st.button("Reset Tool"):
            st.session_state.clear()
            st.rerun()

    # Process input documents
    if start_processing:
        docs = []

        if raw_links:
            urls = [u.strip() for u in raw_links.splitlines() if u.strip()]
            url_docs = fetch_and_split_documents(urls)
            if url_docs:
                docs.extend(url_docs)

        if url_file:
            pdf_docs = read_uploaded_pdf(url_file)
            if pdf_docs:
                splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
                docs.extend(splitter.split_documents(pdf_docs))

        if not docs:
            st.warning("⚠ No text found.")
        else:
            with st.spinner("Building FAISS index..."):
                st.session_state.vector_store = store_in_faiss(docs)

    # Summary
    if st.session_state.vector_store and st.button("Generate Summary"):
        with st.spinner("Summarizing..."):
            st.session_state.summary_data = summarize_sections(st.session_state.vector_store)
            st.session_state.summary_generated = True

    if st.session_state.summary_generated:
        sd = st.session_state.summary_data

        st.subheader("Scheme Summary")
        st.expander("📌 Benefits", expanded=True).write(sd["Benefits"])
        st.expander("📌 Process").write(sd["Process"])
        st.expander("📌 Eligibility").write(sd["Eligibility"])
        st.expander("📌 Documents Required").write(sd["Documents"])

    # Q&A
    if st.session_state.vector_store:
        st.subheader("Ask a Question About the Scheme")
        question = st.text_input("Your question:")

        if question:
            llm = get_cached_llm(st.session_state["selected_model"], GROQ_API_KEY)

            with st.spinner("Generating answer..."):
                answer, docs = run_retrieval_query(llm, st.session_state.vector_store, question)

            sources = []
            for d in docs:
                if "source" in d.metadata:
                    sources.append(d.metadata["source"])

            st.session_state.qa_history.append({
                "question": question,
                "answer": answer,
                "sources": sources
            })

            st.rerun()

        for idx, q in enumerate(st.session_state.qa_history):
            with st.expander(f"Q{idx+1}: {q['question']}"):
                st.markdown(f"<div class='question-box'>{q['answer']}</div>", unsafe_allow_html=True)

                if q["sources"]:
                    st.write("🔗 Sources:")
                    for s in q["sources"]:
                        st.markdown(f"- [{s}]({s})")

# --- Run App ---
if __name__ == "__main__":
    run_app()
