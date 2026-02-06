import warnings
warnings.filterwarnings('ignore', message='.*torch.classes.*')
warnings.filterwarnings('ignore', message='.*resume_download.*')
import os
import streamlit as st
from io import BytesIO
import requests
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime, timezone
from pathlib import Path
import re

# Try pdfplumber first, fall back to pypdf
try:
    import pdfplumber
    USE_PDFPLUMBER = True
except ImportError:
    from pypdf import PdfReader
    USE_PDFPLUMBER = False
    st.warning("⚠️ Install pdfplumber for better table extraction: `pip install pdfplumber`")

# ---------------------------
# Settings
# ---------------------------
st.set_page_config(
    page_title="BoardGame Guru",
    page_icon="assets/images/guru_logo.png" if os.path.exists("assets/images/guru_logo.png") else "🎲",
    layout="centered"
)
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------
# Custom CSS (keeping your original styling)
# ---------------------------
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] { background-color: #2D1940; color: #FFB703; }
    div[data-testid="stTextInput"] > div > input { background-color: #3F00DE !important; color: white !important; }
    textarea { background-color: #531DB3 !important; color: white !important; }
    textarea::placeholder { color: #9F8DB0 !important; opacity: 0.8 !important; }
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
    section[data-testid="stSidebar"] div.stButton > button {
        background-color: #D13B3B !important; color: #FAFAFA !important;
        border-radius: 8px !important; border: 2px solid #8B0000 !important;
        font-weight: bold !important; font-size: 16px !important;
    }
    div[data-testid="stButton"] > button {
        background-color: #569958 !important; color: white !important; 
        border-radius: 8px !important; height: 42px !important;
        font-weight: bold !important; font-size: 16px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Session state
# ---------------------------
for key in ["messages", "last_uploaded_files", "game_name", "file_uploader_key", "pdfs_processed"]:
    if key not in st.session_state:
        if key == "file_uploader_key":
            st.session_state[key] = 0
        elif key == "pdfs_processed":
            st.session_state[key] = False
        else:
            st.session_state[key] = [] if key.endswith("files") or key == "messages" else ""

# ---------------------------
# App header
# ---------------------------
col1, col2 = st.columns([6, 23])
with col1:
    if os.path.exists("assets/images/guru_logo.png"):
        st.image("assets/images/guru_logo.png", width=120)
with col2:
    st.markdown("<h1 style='color:#FAFAFA; margin-top: 15px;'>BoardGame Guru v3</h1>", unsafe_allow_html=True)

st.write(f"Upload board game rulebooks in PDF format. Using: **{'pdfplumber' if USE_PDFPLUMBER else 'pypdf'}**")

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
# Sidebar
# ---------------------------
with st.sidebar:
    if st.session_state.game_name:
        st.markdown(f"<h3 style='color:#FFB703;'>{st.session_state.game_name}</h3>", unsafe_allow_html=True)
    st.markdown("---")
    if st.button("🧹 Reset Chat"):
        for key in ["messages", "last_uploaded_files", "game_name"]:
            st.session_state[key] = [] if isinstance(st.session_state[key], list) else ""
        st.session_state.file_uploader_key += 1
        st.session_state.pdfs_processed = False
        for key in ["model", "index", "all_chunks"]:
            st.session_state.pop(key, None)
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# Token tracking
TOKEN_FILE = Path("daily_tokens.json")
MAX_TOKENS_PER_DAY = 200_000

def load_daily_tokens():
    today = datetime.now(timezone.utc).date().isoformat()
    if TOKEN_FILE.exists():
        data = json.loads(TOKEN_FILE.read_text())
        if data.get("date") == today:
            return data.get("tokens", 0)
    return 0

def save_daily_tokens(tokens):
    today = datetime.now(timezone.utc).date().isoformat()
    TOKEN_FILE.write_text(json.dumps({"date": today, "tokens": tokens}))

with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Daily Token Usage")
    current_tokens = load_daily_tokens()
    st.progress(min(current_tokens / MAX_TOKENS_PER_DAY, 1.0), 
                text=f"{current_tokens / MAX_TOKENS_PER_DAY * 100:.1f}%")

# ---------------------------
# Groq API
# ---------------------------
GROQ_API_KEY = st.secrets["groq"]["api_key"]
GROQ_API_URL = "https://api.groq.com/openai/v1/responses"
GROQ_MODEL = "openai/gpt-oss-120b"

def groq_generate(prompt, max_tokens=2000, temperature=0):
    try:
        response = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": GROQ_MODEL, "input": prompt, "max_output_tokens": max_tokens, "temperature": temperature}
        )
        
        if response.status_code != 200:
            return f"⚠️ API Error ({response.status_code})"
        
        result = response.json()
        
        # Track tokens
        limit = int(response.headers.get("x-ratelimit-limit-tokens", 0))
        remaining = int(response.headers.get("x-ratelimit-remaining-tokens", 0))
        save_daily_tokens(load_daily_tokens() + (limit - remaining))
        
        # Parse response
        if "output_text" in result:
            return result["output_text"].strip()
        if "choices" in result:
            return result["choices"][0].get("message", {}).get("content", "").strip()
        
        return "⚠️ Unexpected response format"
    except Exception as e:
        return f"❌ Error: {e}"

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
# Process PDFs
# ---------------------------
if st.button("⚙️ Process PDFs"):
    current_files = [f.name for f in uploaded_files]
    if current_files != st.session_state.last_uploaded_files:
        st.session_state.messages = []
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.last_uploaded_files = current_files

    @st.cache_data
    def extract_pdf_text(file_data):
        """Extract text using pdfplumber or pypdf."""
        all_text = []
        
        for file_name, file_content in file_data:
            if USE_PDFPLUMBER:
                # Better table handling
                with pdfplumber.open(BytesIO(file_content)) as pdf:
                    for page_num, page in enumerate(pdf.pages, 1):
                        text = page.extract_text() or ""
                        # Extract tables separately and format them
                        tables = page.extract_tables()
                        if tables:
                            text += "\n\n[TABLES ON THIS PAGE]:\n"
                            for table in tables:
                                # Format table as text
                                for row in table:
                                    text += " | ".join(str(cell) if cell else "" for cell in row) + "\n"
                        
                        all_text.append({
                            'file': file_name,
                            'page': page_num,
                            'text': text
                        })
            else:
                # Fallback to pypdf
                reader = PdfReader(BytesIO(file_content))
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text() or ""
                    all_text.append({
                        'file': file_name,
                        'page': page_num,
                        'text': text
                    })
        
        return all_text

    file_data = tuple((f.name, f.getvalue()) for f in uploaded_files)
    pages = extract_pdf_text(file_data)

    # Smaller, smarter chunking
    def create_chunks(pages, chunk_size=500, overlap=100):
        chunks = []
        for page_data in pages:
            text = page_data['text']
            # Clean text
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Detect section headers (ALL CAPS lines)
            header = None
            lines = text.split('\n')
            for line in lines[:5]:
                line = line.strip()
                if line and line.isupper() and 5 < len(line) < 80:
                    header = line
                    break
            
            # Create chunks
            if len(text) < chunk_size:
                chunks.append({
                    'text': text,
                    'page': page_data['page'],
                    'file': page_data['file'],
                    'header': header
                })
            else:
                start = 0
                while start < len(text):
                    end = min(start + chunk_size, len(text))
                    chunk_text = text[start:end]
                    
                    # Prepend header to first chunk
                    if start == 0 and header:
                        chunk_text = f"[{header}]\n{chunk_text}"
                    
                    chunks.append({
                        'text': chunk_text,
                        'page': page_data['page'],
                        'file': page_data['file'],
                        'header': header
                    })
                    start += chunk_size - overlap
        
        return chunks

    all_chunks = create_chunks(pages)

    @st.cache_resource
    def build_index(chunk_texts):
        # Use faster, smaller model for speed
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(chunk_texts, show_progress_bar=True)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index, model

    chunk_texts = [c['text'] for c in all_chunks]
    index, model = build_index(chunk_texts)

    st.session_state.index = index
    st.session_state.model = model
    st.session_state.all_chunks = all_chunks
    st.session_state.pdfs_processed = True

    st.success(f"✅ Processed {len(pages)} pages → {len(all_chunks)} chunks")

if not st.session_state.get("pdfs_processed", False):
    st.stop()

st.markdown("<hr style='border:2px solid cyan;'>", unsafe_allow_html=True)
st.markdown("<h3 style='color:#00FFFF;'>💬 Chat with the Guru</h3>", unsafe_allow_html=True)

# ---------------------------
# Chat
# ---------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("Ask a question about the rules:")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # Simple but effective retrieval
    query_variations = [
        query,
        query.upper(),
        query.lower(),
        # Extract keywords
        " ".join([w for w in query.lower().split() if len(w) > 3])
    ]
    
    # Get more chunks
    all_indices = set()
    for q in query_variations:
        vec = st.session_state.model.encode([q])
        _, indices = st.session_state.index.search(vec, 20)
        all_indices.update(indices[0].tolist())
    
    # Get chunks with metadata
    retrieved = []
    for idx in list(all_indices)[:15]:
        chunk = st.session_state.all_chunks[idx]
        retrieved.append(chunk)
    
    # Sort by header match
    query_keywords = set(query.lower().split())
    retrieved.sort(key=lambda c: (
        not (c.get('header') and any(k in c['header'].lower() for k in query_keywords)),
        -len(c['text'])
    ))
    
    # Format context
    context_parts = []
    for chunk in retrieved[:10]:
        header = f"[{chunk['header']}]" if chunk.get('header') else ""
        page = f"(Page {chunk['page']})"
        context_parts.append(f"{header} {page}\n{chunk['text']}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Debug
    with st.expander("🔍 Retrieved Context"):
        st.text(context[:2000] + "..." if len(context) > 2000 else context)
    
    # Generate answer
    prompt = f"""You are a board game rules expert.

Rulebook excerpts (with section headers and page numbers):

{context}

User question: {query}

Answer using ONLY the rulebook excerpts above. Pay attention to [SECTION] headers. 
Include page numbers when citing rules. If not found, say so.
"""

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = groq_generate(prompt, max_tokens=2000)
            st.write(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})