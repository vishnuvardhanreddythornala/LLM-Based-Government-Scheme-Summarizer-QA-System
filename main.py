# --- Import necessary libraries ---
import configparser  # For reading API key from config file
import streamlit as st  # Streamlit to build the web interface
import pickle  # For saving and loading FAISS index (no longer used directly here)
import logging  # To log application activity and errors
from datetime import datetime  # For timestamped logging
import os  # For managing files and directories

# LangChain and HuggingFace components for document loading and question answering
from langchain_community.document_loaders import UnstructuredURLLoader  # To load content from URLs
from langchain.text_splitter import RecursiveCharacterTextSplitter  # To break large text into smaller chunks
from langchain_huggingface import HuggingFaceEmbeddings  # For generating vector embeddings
from langchain_community.vectorstores import FAISS  # FAISS for storing and searching vectors
from langchain_groq import ChatGroq  # Interface to use Groq's LLM
from langchain.chains import RetrievalQA  # QA chain using retrieval + LLM
from langchain_community.document_loaders import PyPDFLoader  # To load PDF files

from streamlit.components.v1 import html  # For custom HTML like JS injection

# Set page config for wide layout
st.set_page_config(page_title="Scheme Research Tool", layout="wide")

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
    width: auto !important;  /* Fix full width issue */
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

/*  Inputs */
input, textarea {
    background-color: #0d1117 !important;
    color: white !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}

/*  Question Answer Section */
.question-box {
    font-size: 1.15rem;
    line-height: 1.7rem;
    margin-bottom: 1rem;
    color: #c9d1d9;
}

/*  Source Link */
a {
    color: #58a6ff !important;
    text-decoration: underline;
}

/*  Headings */
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
 /* fixed length keeps it always centered */
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

# --- Load API key from .config file ---
import os
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")




@st.cache_resource(show_spinner=False)
def get_cached_llm(model_name: str, api_key: str):
    return ChatGroq(model=model_name, api_key=api_key)



# --- Setup Logging ---
log_filename = f"logs/scheme_tool_{datetime.now().strftime('%Y-%m-%d')}.log"
os.makedirs("logs", exist_ok=True)  # Create log directory if it doesn't exist
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





# --- Check whether the API key is available ---
def check_api_key():
    if not GROQ_API_KEY:
        logging.error("Groq API key missing. Cannot proceed.")
        st.error("Missing Groq API Key. Please add it to .config.")
        st.stop()
   

# --- Extract content from uploaded PDF file ---
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

# --- Load and split document content into smaller chunks for embedding ---
def fetch_and_split_documents(url_list):
    try:
        with st.spinner("Fetching content from URLs..."):
            loader = UnstructuredURLLoader(urls=url_list)
            raw_docs = loader.load()
            if not raw_docs or all(not doc.page_content.strip() for doc in raw_docs):
                logging.warning("No usable content found.")
                return None
            splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
            split_docs = splitter.split_documents(raw_docs)
            logging.info(f"Split into {len(split_docs)} chunks from URLs.")
            return split_docs
    except Exception as e:
        logging.error(f"Error loading documents: {e}")
        st.error(f"Error loading content: {e}")
        return None

# --- Create a FAISS index from document chunks and save it using FAISS method ---
def store_in_faiss(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    vector_index = FAISS.from_documents(text_chunks, embeddings)
    vector_index.save_local("faiss_store_openai")
    logging.info("FAISS vector store created and saved using save_local().")
    return vector_index

# --- Generate summarized responses for predefined questions ---
def summarize_sections(index):
    groq_llm = get_cached_llm(st.session_state["selected_model"], GROQ_API_KEY)

    questions = {
        "Benefits": "Summarize the key benefits of the scheme.",
        "Process": "Describe the application process for the scheme.",
        "Eligibility": "What are the eligibility criteria for this scheme?",
        "Documents": "List documents required to apply."
    }
    summary = {}
    for key, prompt in questions.items():
        qa_pipeline = RetrievalQA.from_chain_type(
            llm=groq_llm,
            chain_type="stuff",
            retriever=index.as_retriever(search_kwargs={"k": 10}),
            return_source_documents=True
        )
        result = qa_pipeline.invoke(prompt)
        summary[key] = result["result"].strip() or "No information found."
    return summary

# --- Main Streamlit app logic ---
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

    # Inject CSS for button styling
    button_style = """
    <style>
    div.stButton > button:first-child {
        background-color: #238636;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        height: 3em;
        width: 100%;
        border: none;
        transition: 0.3s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        background-color: #2ea043;
        transform: scale(1.02);
    }
    </style>
    """
    st.markdown(button_style, unsafe_allow_html=True)


    st.session_state.setdefault("qa_history", [])
    st.session_state.setdefault("summary_generated", False)
    st.session_state.setdefault("vector_store", None)
    st.session_state.setdefault("summary_data", None)



    

    with st.sidebar:
        st.header("Select Input Type")

        input_type = st.radio(
            "Choose input method:",
            ("None", "URLs", "PDF"),
            index=0
        )

        if input_type == "URLs":
            st.subheader("🔗 Enter Scheme URLs")
            raw_links = st.text_area("Paste one URL per line")
            url_file = None

        elif input_type == "PDF":
            st.subheader("Upload PDF File")
            url_file = st.file_uploader("",type=["pdf"])
            raw_links = None

        else:
            raw_links = None
            url_file = None

        #  NEW: Add model selector here
        model_choice = st.radio(
            "Choose Model:",
            ["🟢 llama3-8b (Fast)", "🔵 llama3-70b (Accurate)"],
            index=0
        )
        model_map = {
            "🟢 llama3-8b (Fast)": "llama3-8b-8192",
            "🔵 llama3-70b (Accurate)": "llama3-70b-8192"
        }
        st.session_state["selected_model"] = model_map[model_choice]

        if "model_logged" not in st.session_state:
            
            st.session_state.model_logged = True



        st.markdown("---")
        start_processing = st.button("Process")
        if st.button("Reset Tool"):
            # Clear session state
            for key in st.session_state.keys():
                del st.session_state[key]
            logging.info("User triggered Reset Tool.")
            # Optional: remove uploaded files if any
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
            
            with st.spinner("⚡Powering up your scheme assistant..."):
                vector_db = store_in_faiss(all_documents)
                st.session_state.vector_store = vector_db
        else:
            st.warning("⚠ Could not extract usable text from the given inputs.")
            logging.warning("No documents found to process.")

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


                        qa_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            chain_type="stuff",
                            retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": 10}),
                            return_source_documents=True
                        )
                        result = qa_chain.invoke(question)
                        answer = result["result"]
                        logging.info(f"Answer generated for question: {question}")
                        if len(answer.strip()) < 10:
                            logging.warning(f"Short or empty answer returned for: {question}")

                        source_urls = set()
                        for doc in result.get("source_documents", []):
                            metadata = getattr(doc, "metadata", {})
                            if "source" in metadata:
                                source_urls.add(metadata["source"])

                        st.session_state.qa_history.append({
                            "question": question,
                            "answer": answer,
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

                        #  Inline web URLs
                        if urls:
                            joined_urls = " ".join(f"<a href='{u}' target='_blank'>{u}</a>" for u in urls)
                            st.markdown(f"<p>🔗 <b>Source URLs:</b> {joined_urls}</p>", unsafe_allow_html=True)

                        #  Download buttons for PDFs
                        for src in pdfs:
                            file_path = "." + src
                            file_name = os.path.basename(file_path)
                            try:
                                with open(file_path, "rb") as f:
                                    pdf_bytes = f.read()
                                    
                                    st.markdown(
    f"""<p style='font-size: 1.15rem; font-weight: 600; margin-bottom: 0.2rem; color: #c9d1d9;'> 🔗 Source PDFs:</p>""",
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
                        
        
                          

# --- Run the application ---
if __name__ == "__main__":
    run_app()
    

