import streamlit as st
import os
import tempfile
import time
import uuid
import pandas as pd
import plotly.express as px

# ---------------------------------------------------
# HUGGINGFACE CACHE FIX
# ---------------------------------------------------
os.environ["HF_HOME"] = "C:/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "C:/huggingface_cache"

# ---------------------------------------------------
# IMPORTS
# ---------------------------------------------------
from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain.chains.combine_documents import (
    create_stuff_documents_chain
)

from langchain_core.prompts import (
    ChatPromptTemplate
)

from langchain.chains import (
    create_retrieval_chain
)

from langchain.memory import (
    ConversationBufferMemory
)

from langchain_community.vectorstores import (
    FAISS
)

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader
)

from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Document Intelligence Platform",
    page_icon="📄",
    layout="wide"
)


# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>

/* Main Background */
.stApp {
    background: linear-gradient(135deg,#0f172a,#111827);
    color:white;
}

/* Headings */
h1,h2,h3,h4 {
    color:white !important;
}

/* Upload Box */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
}

/* Upload Text */
[data-testid="stFileUploader"] section {
    color:white !important;
}

/* Browse Button */
[data-testid="stFileUploader"] button {
    background: linear-gradient(to right,#2563eb,#7c3aed) !important;
    color:white !important;
    border-radius:10px !important;
    border:none !important;
    font-weight:600 !important;
    transition:0.3s ease;
}

/* Browse Button Hover */
[data-testid="stFileUploader"] button:hover {
    transform: scale(1.03);
    background: linear-gradient(to right,#1d4ed8,#6d28d9) !important;
}

/* Glass Card */
.glass {
    background: rgba(255,255,255,0.05);
    border-radius:18px;
    padding:24px;
    border:1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
    transition:0.3s ease;
}

/* Card Hover */
.glass:hover {
    transform: translateY(-3px);
    box-shadow:0px 8px 20px rgba(0,0,0,0.3);
}

/* Metric Cards */
.metric-card {
    background: rgba(255,255,255,0.06);
    padding:20px;
    border-radius:16px;
    text-align:center;
    border:1px solid rgba(255,255,255,0.08);
}

/* Buttons */
.stButton>button {
    width:100%;
    height:3.2em;
    border-radius:12px;
    background: linear-gradient(to right,#2563eb,#7c3aed);
    color:white;
    border:none;
    font-size:16px;
    font-weight:600;
    transition:0.3s ease;
}

/* Button Hover */
.stButton>button:hover {
    transform: scale(1.02);
    box-shadow:0px 5px 15px rgba(37,99,235,0.4);
}

/* Text Input */
.stTextInput>div>div>input {
    background: rgba(255,255,255,0.05);
    color:white;
    border-radius:12px;
    border:1px solid rgba(255,255,255,0.08);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background:#111827;
}

/* Expander */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.05);
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# LOAD ENV VARIABLES
# ---------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "query_count" not in st.session_state:
    st.session_state.query_count = 0

if "vectors" not in st.session_state:
    st.session_state.vectors = None

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = 0


# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown("""
<div style='text-align:center;padding:25px;'>

<h1 style='font-size:56px;font-weight:700;'>
AI Document Intelligence Platform
</h1>

<p style='font-size:20px;color:#cbd5e1;'>

Upload Documents • Chat with PDFs • Generate Infographics

</p>

</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
with st.sidebar:

    st.markdown("## Settings")

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
        1000
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        0,
        500,
        200
    )

    st.markdown("---")

    st.markdown("## Analytics")

    st.metric(
        "Documents Loaded",
        st.session_state.documents_loaded
    )

    st.metric(
        "Questions Asked",
        st.session_state.query_count
    )


# ---------------------------------------------------
# LLM
# ---------------------------------------------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=selected_model
)


# ---------------------------------------------------
# MEMORY
# ---------------------------------------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)


# ---------------------------------------------------
# PROMPT TEMPLATE
# ---------------------------------------------------
prompt = ChatPromptTemplate.from_template(
    """
    You are an advanced AI assistant.

    Answer ONLY from the provided context.

    If answer is unavailable,
    say:
    'The uploaded documents do not contain this information.'

    Context:
    {context}

    Question:
    {input}
    """
)


# ---------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------
left_col, right_col = st.columns([2.5,1])


# ---------------------------------------------------
# FILE UPLOAD SECTION
# ---------------------------------------------------
with left_col:

    st.markdown("""
    <div class='glass'>

    <h3>Upload Documents</h3>

    <p style='color:#cbd5e1;'>
    Supported formats: PDF, TXT, CSV
    </p>

    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload Files",
        type=["pdf","txt","csv"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )


# ---------------------------------------------------
# DASHBOARD SECTION
# ---------------------------------------------------
with right_col:

    total_files = len(uploaded_files) if uploaded_files else 0

    st.markdown(f"""
    <div class='metric-card'>

    <h2>{total_files}</h2>

    <p>Files Uploaded</p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class='glass'>

    <h4>Features</h4>

    <ul>
    <li>AI Chat with PDFs</li>
    <li>Generate Infographics</li>
    <li>Semantic Search</li>
    <li>Document Analytics</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------
# LOAD DOCUMENTS
# ---------------------------------------------------
def load_documents(uploaded_files):

    documents = []

    for uploaded_file in uploaded_files:

        file_extension = uploaded_file.name.split(".")[-1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f".{file_extension}"
        ) as temp_file:

            temp_file.write(uploaded_file.read())

            temp_path = temp_file.name

        if file_extension == "pdf":

            loader = PyPDFLoader(temp_path)

        elif file_extension == "txt":

            loader = TextLoader(temp_path)

        elif file_extension == "csv":

            loader = CSVLoader(temp_path)

        docs = loader.load()

        documents.extend(docs)

    return documents


# ---------------------------------------------------
# VECTOR EMBEDDING
# ---------------------------------------------------
def create_vector_embedding(uploaded_files):

    documents = load_documents(uploaded_files)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    final_documents = text_splitter.split_documents(
        documents
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="C:/huggingface_cache"
    )

    vectors = FAISS.from_documents(
        final_documents,
        embeddings
    )

    vector_db_path = f"vectorstore_{uuid.uuid4()}"

    vectors.save_local(vector_db_path)

    return vectors, len(documents)


# ---------------------------------------------------
# CREATE VECTOR DB
# ---------------------------------------------------
st.markdown("---")

if st.button("Create Vector Database"):

    if uploaded_files:

        with st.spinner("Creating Vector Database..."):

            progress = st.progress(0)

            for i in range(100):
                time.sleep(0.01)
                progress.progress(i + 1)

            vectors, document_count = create_vector_embedding(
                uploaded_files
            )

            st.session_state.vectors = vectors

            st.session_state.documents_loaded = document_count

        st.success("Vector Database Created Successfully")

    else:

        st.warning("Please upload files first")


# ---------------------------------------------------
# QUESTION INPUT
# ---------------------------------------------------
st.markdown("---")

prompt1 = st.text_input(
    "Ask Questions",
    placeholder="""
Examples:
- Summarize this document
- Generate infographic
- Create flowchart
- Explain chapter 2
"""
)


# ---------------------------------------------------
# RESPONSE SECTION
# ---------------------------------------------------
if prompt1:

    st.session_state.query_count += 1

    if st.session_state.vectors is None:

        st.warning("Please create vector database first")

    else:

        with st.spinner("Generating Response..."):

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

        query = prompt1.lower()

        # ---------------------------------------------------
        # INFOGRAPHIC MODE
        # ---------------------------------------------------
        if any(word in query for word in [
            "infographic",
            "visual summary",
            "diagram",
            "mind map",
            "flowchart"
        ]):

            infographic_prompt = f"""
            Create a professional infographic-style summary.

            Content:
            {response['answer']}

            Include:
            - Title
            - Key Concepts
            - Important Insights
            - Conclusion
            """

            infographic_response = llm.invoke(
                infographic_prompt
            )

            st.markdown("## Infographic")

            st.markdown(f"""
            <div class='glass'>

            {infographic_response.content}

            </div>
            """, unsafe_allow_html=True)

        # ---------------------------------------------------
        # NORMAL RESPONSE
        # ---------------------------------------------------
        else:

            st.markdown("## AI Response")

            st.markdown(f"""
            <div class='glass'>

            {response['answer']}

            </div>
            """, unsafe_allow_html=True)

        # ---------------------------------------------------
        # CHAT HISTORY
        # ---------------------------------------------------
        st.session_state.chat_history.append({
            "question": prompt1,
            "answer": response['answer']
        })

        # ---------------------------------------------------
        # METRICS
        # ---------------------------------------------------
        col1, col2 = st.columns(2)

        with col1:

            st.info(
                f"Response Time: {end-start:.2f} sec"
            )

        with col2:

            st.success(
                f"Retrieved Chunks: {len(response['context'])}"
            )

        # ---------------------------------------------------
        # SOURCES
        # ---------------------------------------------------
        with st.expander("Retrieved Chunks with Sources"):

            for i, doc in enumerate(response["context"]):

                page_number = doc.metadata.get(
                    "page",
                    "N/A"
                )

                source = doc.metadata.get(
                    "source",
                    "Unknown"
                )

                st.markdown(f"""
                ### Chunk {i+1}

                Source: {source}

                Page: {page_number}
                """)

                st.write(doc.page_content)

                st.markdown("---")

        # ---------------------------------------------------
        # EXPORT
        # ---------------------------------------------------
        st.download_button(
            "Download Response",
            response['answer'],
            file_name="summary.txt"
        )


# ---------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------
if st.session_state.chat_history:

    st.markdown("---")

    st.markdown("## Chat History")

    for item in reversed(
        st.session_state.chat_history
    ):

        st.markdown(f"""
        <div class='glass'>

        <b>Question:</b><br>
        {item['question']}

        <br><br>

        <b>Answer:</b><br>
        {item['answer']}

        </div>

        <br>
        """, unsafe_allow_html=True)


# ---------------------------------------------------
# ANALYTICS DASHBOARD
# ---------------------------------------------------
st.markdown("---")

st.markdown("## Analytics Dashboard")

analytics_df = pd.DataFrame({
    "Metric":[
        "Questions Asked",
        "Documents Loaded"
    ],
    "Value":[
        st.session_state.query_count,
        st.session_state.documents_loaded
    ]
})

fig = px.bar(
    analytics_df,
    x="Metric",
    y="Value",
    title="Platform Analytics"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")

st.markdown("""
<div style='text-align:center;color:gray;'>

Built with Streamlit • LangChain • Groq • FAISS • HuggingFace

</div>
""", unsafe_allow_html=True)
