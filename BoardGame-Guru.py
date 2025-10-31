import os
import streamlit as st
from io import BytesIO
from pypdf import PdfReader
import requests
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------
# Settings
# ---------------------------
st.set_page_config(
    page_title="BoardGame Guru",
    page_icon="assets/images/guru_logo.png",  
    layout="centered"
)
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)





# ---------------------------
# Custom CSS for styling
# ---------------------------
st.markdown(
    """
    <style>
    /* Sidebar background */
    section[data-testid="stSidebar"] { background-color: #2D1940; color: #FFB703; }

    /* Board Game Name input */
    div[data-testid="stTextInput"] > div > input { 
        background-color: #3F00DE !important; 
        color: white !important; 
    }

    
    /* User Chat Input field (at the bottom) */
    div[data-testid="stChatInput"] input {
        background-color: #4C00E0 !important; /* Example: A new color for the chat box */
        color: white !important;             /* Keep text color readable */
    }



    /* Drag & drop uploader */
    div[data-testid="stFileUploader"] > section {
        background-color: #E3B646 !important;
        border: 3px dashed black !important;
        border-radius: 15px !important;
        color: black !important;
        padding: 50px !important;
        min-height: 140px !important;
        font-size: 18px !important;
        text-align: center !important;
    }
    div[data-testid="stFileUploader"] > section * { color: black !important; font-weight: bold !important; }
    div[data-testid="stFileUploader"] section button {
        background-color: #08010D !important; color: #E3B646 !important;
        border: 2px solid #E3B646 !important; border-radius: 8px !important;
        padding: 8px 16px !important; font-weight: bold !important; font-size: 16px !important; cursor: pointer !important;
    }
    div[data-testid="stFileUploader"] section button:hover { background-color: #000000 !important; color: #FAFAFA !important; }

    .stAlert.stAlert-info { background-color: #2C2C3C !important; color: #FAFAFA !important; border: 1px solid #FFB703 !important; }

    /* Reset Chat button (sidebar) */
    section[data-testid="stSidebar"] div.stButton > button {
        background-color: #D62F2F !important; color: #FAFAFA !important;
        border-radius: 8px !important; border: 2px solid #8B0000 !important;
        font-weight: bold !important; font-size: 16px !important; width: 160px !important; height: 42px !important;
        margin-top: 10px !important; transition: all 0.2s ease-in-out !important;
    }
    section[data-testid="stSidebar"] div.stButton > button:hover { background-color: #DE0202 !important; color: white !important; transform: scale(1.05); }

    button[title="Close sidebar"], button[title="Open sidebar"] { background-color: transparent !important; border: none !important; color: inherit !important; }

    /* Process PDFs button (main) */
    div[data-testid="stButton"] > button {
        background-color: #358239 !important; color: white !important; 
        border-radius: 8px !important; height: 42px !important; width: 160px !important;
        font-weight: bold !important; font-size: 16px !important; border: 2px solid #2E7D32 !important;
        transition: all 0.2s ease-in-out !important;
    }
    div[data-testid="stButton"] > button:hover { background-color: #027300 !important; transform: scale(1.05); }

    /* Sticky game name */
    .sticky-game-name { position: sticky; top: 0; background-color:#08010D; color:#FFB703; padding:5px; z-index:1000; font-size: 40px; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------
# Session state defaults
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_uploaded_files" not in st.session_state:
    st.session_state.last_uploaded_files = []

if "game_name" not in st.session_state:
    st.session_state.game_name = ""

if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0  # used to reset uploader

if "pdfs_processed" not in st.session_state:
    st.session_state.pdfs_processed = False

# ---------------------------
# App header
# ---------------------------
col1, col2 = st.columns([6, 23])
with col1:
    st.image("assets/images/guru_logo.png", width=120)
with col2:
    st.markdown("<h1 style='color:#FAFAFA; margin-top: 15px;'>BoardGame Guru</h1>", unsafe_allow_html=True)

st.write("Upload board game rulebooks in PDF format and ask questions about the rules!")


# ---------------------------
# Game Name Input
# ---------------------------
game_name_input = st.text_input(
    "Board Game Name:",
    value=st.session_state.game_name,
    key=f"game_name_input_{st.session_state.file_uploader_key}"
)
if game_name_input:
    st.session_state.game_name = game_name_input




# ---------------------------
# Sidebar: Reset button & Game Name display
# ---------------------------
with st.sidebar:
    if st.session_state.game_name:
        st.markdown(f"<h3 style='color:#FFB703; font-size:20px; font-family:Comic Sans MS;'>{st.session_state.game_name}</h3>", 
                    unsafe_allow_html=True)

    if st.button("🧹 Reset Chat", key="reset"):
        st.session_state.messages = []
        st.session_state.last_uploaded_files = []
        st.session_state.game_name = ""
        st.session_state.file_uploader_key += 1
        st.session_state.pdfs_processed = False
        st.session_state.pop("model", None)
        st.session_state.pop("index", None)
        st.session_state.pop("embeddings", None)
        st.session_state.pop("all_chunks", None)
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# ---------------------------
# Sidebar: Buy Me a Coffee
# ---------------------------

with st.sidebar:
    # Separator
    st.markdown("---")
    
    # Styled text
    st.markdown(
        """
        <p style="color:#FCF2D9; font-size:16px;">
        💰 Support me!<br>
        If you enjoy this app, consider buying me a coffee! Your support helps me maintain and improve the app.
        </p>
        """,
        unsafe_allow_html=True
    )

    # Styled button with hover effect
    st.markdown(
        """
        <style>
        .bmc-button {
            background-color:#176396;
            color:white;
            border:none;
            border-radius:8px;
            padding:10px 20px;
            font-size:16px;
            font-weight:bold;
            cursor:pointer;
            margin-top:5px;
            transition: all 0.3s ease;
        }
        .bmc-button:hover {
            background-color:#000CFF;
            transform: scale(1.05);
        }
        </style>
        <a href="https://buymeacoffee.com/vasileios" target="_blank">
            <button class="bmc-button">☕ Buy Me a Coffee</button>
        </a>
        """,
        unsafe_allow_html=True
    )



# ---------------------------
# Groq API setup
# ---------------------------
GROQ_API_KEY = st.secrets["groq"]["api_key"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
GROQ_API_URL = "https://api.groq.com/openai/v1/responses"
GROQ_MODEL = "openai/gpt-oss-120b"

def groq_generate(prompt, max_tokens=250, temperature=0):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": GROQ_MODEL, "input": prompt, "max_output_tokens": max_tokens, "temperature": temperature}
    response = requests.post(GROQ_API_URL, headers=headers, data=json.dumps(payload))
    if response.status_code != 200:
        return f"API Error ({response.status_code}): {response.text}"
    result = response.json()
    if "output_text" in result and result["output_text"]:
        return result["output_text"].strip()
    if "choices" in result:
        for choice in result["choices"]:
            content = choice.get("message", {}).get("content")
            if content:
                return content.strip()
    if "output" in result and len(result["output"]) > 0:
        for item in result["output"]:
            if "content" in item:
                for c in item["content"]:
                    if c.get("type") == "output_text" and c.get("text"):
                        return c["text"].strip()
    return "Unexpected response format from Groq."

# ---------------------------
# File uploader
# ---------------------------
uploaded_files = st.file_uploader(
    "Upload one or more rulebook PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    key=f"file_uploader_{st.session_state.file_uploader_key}"
)
if not uploaded_files:
    st.info("Please upload one or more PDF rulebooks to continue.")
    st.stop()

# ---------------------------
# Process PDFs button
# ---------------------------
if st.button("⚙️ Process PDFs"):
    current_files = [f.name for f in uploaded_files]
    if current_files != st.session_state.last_uploaded_files:
        st.session_state.messages = []
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.last_uploaded_files = current_files
        st.toast("🔄 New PDF(s) detected — cache and chat history cleared.", icon="🔁")

    @st.cache_data
    def extract_pdf_texts(file_data):
        pdf_texts = []
        for file_name, file_content in file_data:
            reader = PdfReader(BytesIO(file_content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            pdf_texts.append((file_name, text))
        return pdf_texts

    file_data = tuple((f.name, f.getvalue()) for f in uploaded_files)
    pdf_texts = extract_pdf_texts(file_data)

    def chunk_text(text, chunk_size=1200, overlap=200):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    all_chunks = []
    for name, text in pdf_texts:
        chunks = chunk_text(text)
        all_chunks.extend(chunks)

    @st.cache_resource
    def build_faiss_index(chunks):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index, model, embeddings

    index, model, embeddings = build_faiss_index(all_chunks)

    st.session_state.index = index
    st.session_state.model = model
    st.session_state.embeddings = embeddings
    st.session_state.all_chunks = all_chunks
    st.session_state.pdfs_processed = True
    st.session_state.index_ready = True

    # ✅ Store messages only (don't display yet)
    st.session_state.pdf_messages = [
        f"✅ Loaded {len(pdf_texts)} PDF(s) successfully",
        f"✅ Indexed {len(all_chunks)} text chunks for retrieval"
    ]


    # # Display messages now
    # for msg in st.session_state.pdf_messages:
    #     st.success(msg)


# ---------------------------
# Stop until PDFs are processed
# ---------------------------
if not st.session_state.get("pdfs_processed", False):
    st.stop()

# Redisplay success messages after re-run
if "pdf_messages" in st.session_state:
    for msg in st.session_state.pdf_messages:
        st.success(msg)



# ---------------------------
# Visual separator before chat section
# ---------------------------
st.markdown("<hr style='border:2px solid cyan; margin-top:30px; margin-bottom:30px;'>", unsafe_allow_html=True)
st.markdown("<h3 style='color:#00FFFF;'>💬 Chat with the Guru</h3>", unsafe_allow_html=True)




# ---------------------------
# Chat history
# ---------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------------------
# Chat input
# ---------------------------
query = st.chat_input("Ask a question about the rules:")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # ---- RAG retrieval ----
    query_vec = st.session_state.model.encode([query], convert_to_numpy=True)
    top_k = 10
    distances, indices = st.session_state.index.search(query_vec, top_k)
    retrieved_chunks = [st.session_state.all_chunks[i] for i in indices[0]]
    retrieved_text = "\n\n".join(retrieved_chunks)

    # ---- Prompt construction ----
    recent_history = "\n".join(
        [f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages[-3:]]
    )

    prompt = f"""
    You are a board game rules expert.

    Here are the most relevant excerpts from the rulebook:

    {retrieved_text}

    Recent conversation (for reference only — ignore if unrelated):
    {recent_history}

    User's question: {query}

    Answer clearly and concisely, using only information from the rulebook.
    If the answer isn't found in the rulebook, reply: "That information cannot be found in the provided pdfs."
    """


    # ---- Generate answer ----
    with st.chat_message("assistant"):
        with st.spinner("Meditating..."):
            answer = groq_generate(prompt, max_tokens=3000, temperature=0.3)
            st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
