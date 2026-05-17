
import streamlit as st
import os
import tempfile
import time

# HuggingFace cache fix
os.environ["HF_HOME"] = "C:/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "C:/huggingface_cache"

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown(
    """
    <style>

    .main {
        background-color: #0E1117;
    }

    .stApp {
        background: linear-gradient(to bottom right, #0f172a, #111827);
        color: white;
    }

    h1, h2, h3 {
        color: white !important;
    }

    .glass {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        padding: 20px;
        border-radius: 16px;
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }

    .metric-card {
        background: rgba(255,255,255,0.08);
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background: linear-gradient(to right, #2563eb, #7c3aed);
        color: white;
        border: none;
        font-weight: 600;
        font-size: 16px;
    }

    .stTextInput>div>div>input {
        background-color: rgba(255,255,255,0.08);
        color: white;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
    }

    .uploadedFile {
        background-color: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# LOAD ENV VARIABLES
# -----------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    """
    <div style='text-align:center;padding:20px;'>
        <h1 style='font-size:52px;'>📄 AI PDF Assistant</h1>
        <p style='font-size:20px;color:#cbd5e1;'>
            Upload PDFs • Create Vector DB • Ask Questions Instantly
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:

    st.markdown("## ⚙️ Settings")

    selected_model = st.selectbox(
        "Choose Groq Model",
        [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768"
        ]
    )

    chunk_size = st.slider(
        "Chunk Size",
        500,
        2000,
        1000,
        100
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        0,
        500,
        200,
        50
    )

    st.markdown("---")

    st.info(
        "Upload PDF files and create a vector database before asking questions."
    )


# -----------------------------
# LLM
# -----------------------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=selected_model
)


# -----------------------------
# PROMPT TEMPLATE
# -----------------------------
prompt = ChatPromptTemplate.from_template(
    """
    You are a professional AI research assistant.

    Answer ONLY from the provided context.

    If the answer is not available in the context,
    say: "The uploaded documents do not contain this information."

    <context>
    {context}
    </context>

    Question: {input}
    """
)


# -----------------------------
# MAIN LAYOUT
# -----------------------------
left_col, right_col = st.columns([2, 1])


# -----------------------------
# LEFT SECTION
# -----------------------------
with left_col:

    st.markdown("### 📂 Upload PDF Documents")

    uploaded_files = st.file_uploader(
        "Drag and drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )


# -----------------------------
# RIGHT SECTION
# -----------------------------
with right_col:

    st.markdown("### 📊 Dashboard")

    total_files = len(uploaded_files) if uploaded_files else 0

    st.markdown(
        f"""
        <div class='metric-card'>
            <h2>{total_files}</h2>
            <p>PDF Files Uploaded</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# -----------------------------
# VECTOR EMBEDDING FUNCTION
# -----------------------------
def create_vector_embedding(uploaded_files):

    documents = []

    for uploaded_file in uploaded_files:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:

            temp_file.write(uploaded_file.read())

            temp_path = temp_file.name

        loader = PyPDFLoader(temp_path)

        docs = loader.load()

        documents.extend(docs)


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )


    final_documents = text_splitter.split_documents(documents)


    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="C:/huggingface_cache"
    )


    vectors = FAISS.from_documents(
        final_documents,
        embeddings
    )

    return vectors


# -----------------------------
# CREATE VECTOR STORE
# -----------------------------
st.markdown("---")

if st.button(" Create Vector Database"):

    if uploaded_files:

        progress_bar = st.progress(0)

        with st.spinner("Processing PDFs and creating embeddings..."):

            for percent_complete in range(100):
                time.sleep(0.01)
                progress_bar.progress(percent_complete + 1)

            st.session_state.vectors = create_vector_embedding(uploaded_files)

        st.success("✅ Vector Database Created Successfully")

    else:
        st.warning("⚠️ Please upload at least one PDF file")


# -----------------------------
# QUESTION INPUT
# -----------------------------
st.markdown("---")

st.markdown("### 💬 Ask Questions")

prompt1 = st.text_input(
    "Enter your question",
    placeholder="Example: Summarize the uploaded document..."
)


# -----------------------------
# RESPONSE GENERATION
# -----------------------------
if prompt1:

    if "vectors" not in st.session_state:

        st.warning("⚠️ Please create vector database first")

    else:

        with st.spinner("Generating response..."):

            document_chain = create_stuff_documents_chain(
                llm,
                prompt
            )

            retriever = st.session_state.vectors.as_retriever()

            retrieval_chain = create_retrieval_chain(
                retriever,
                document_chain
            )

            start = time.process_time()

            response = retrieval_chain.invoke(
                {'input': prompt1}
            )

            end = time.process_time()


        st.markdown("### 🤖 AI Response")

        st.markdown(
            f"""
            <div class='glass'>
                {response['answer']}
            </div>
            """,
            unsafe_allow_html=True
        )


        col1, col2 = st.columns(2)

        with col1:
            st.info(f"⏱️ Response Time: {end - start:.2f} sec")

        with col2:
            st.success(f"📄 Retrieved Chunks: {len(response['context'])}")


        # Similarity Search
        with st.expander("📚 View Retrieved Document Chunks"):

            for i, doc in enumerate(response["context"]):

                st.markdown(f"### Chunk {i+1}")

                st.write(doc.page_content)

                st.markdown("---")


# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")

st.markdown(
    """
    <div style='text-align:center;color:gray;'>
        Built with Streamlit • LangChain • Groq • FAISS • HuggingFace
    </div>
    """,
    unsafe_allow_html=True
)
