import warnings
warnings.filterwarnings('ignore', message='.*torch.classes.*')
warnings.filterwarnings('ignore', message='.*resume_download.*')
import os
import streamlit as st
from io import BytesIO
from pypdf import PdfReader
import requests
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime, timezone
from pathlib import Path
import re
from collections import Counter
import math

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
    div[data-testid="stTextInput"] > div > input { 
        background-color: #3F00DE !important; 
        color: white !important; 
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] p {
        margin-top: 2px !important;
        margin-bottom: 2px !important;
    }
    section[data-testid="stSidebar"] div.stButton > button {
        margin-top: 4px !important;
        margin-bottom: 2px !important;
    }
    section[data-testid="stSidebar"] hr {
        margin-top: 4px !important;
        margin-bottom: 8px !important;
    }
    textarea {
        background-color: #531DB3 !important; 
        color: white !important; 
    }
    textarea::placeholder {
        color: #9F8DB0 !important;
        opacity: 0.8 !important;
    }
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
    section[data-testid="stSidebar"] div.stButton > button {
        background-color: #D13B3B !important; color: #FAFAFA !important;
        border-radius: 8px !important; border: 2px solid #8B0000 !important;
        font-weight: bold !important; font-size: 16px !important; width: 160px !important; height: 42px !important;
        margin-top: 10px !important; transition: all 0.2s ease-in-out !important;
    }
    section[data-testid="stSidebar"] div.stButton > button:hover { background-color: #DE0202 !important; color: white !important; transform: scale(1.05); }
    button[title="Close sidebar"], button[title="Open sidebar"] { background-color: transparent !important; border: none !important; color: inherit !important; }
    div[data-testid="stButton"] > button {
        background-color: #569958 !important; color: white !important; 
        border-radius: 8px !important; height: 42px !important; width: 160px !important;
        font-weight: bold !important; font-size: 16px !important; border: 2px solid #2E7D32 !important;
        transition: all 0.2s ease-in-out !important;
    }
    div[data-testid="stButton"] > button:hover { background-color: #027300 !important; transform: scale(1.05); }
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
    st.session_state.file_uploader_key = 0
if "pdfs_processed" not in st.session_state:
    st.session_state.pdfs_processed = False

# ---------------------------
# App header
# ---------------------------
col1, col2 = st.columns([6, 23])
with col1:
    if os.path.exists("assets/images/guru_logo.png"):
        st.image("assets/images/guru_logo.png", width=120)
    else:
        st.markdown("# 🎲")
with col2:
    st.markdown("<h1 style='color:#FAFAFA; margin-top: 15px;'>BoardGame Guru v2</h1>", unsafe_allow_html=True)

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
# Sidebar
# ---------------------------
with st.sidebar:
    if st.session_state.game_name:
        st.markdown(f"<h3 style='color:#FFB703; font-size:20px; margin-bottom: 2px !important; font-family:Comic Sans MS;'>{st.session_state.game_name}</h3>", 
                    unsafe_allow_html=True)
    st.markdown("---")
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
        st.session_state.pop("chunk_metadata", None)
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# ---------------------------
# Sidebar: Buy Me a Coffee
# ---------------------------
with st.sidebar:
    st.markdown("---")
    st.markdown(
        """
        <p style="color:#FCF2D9; font-size:16px;">
        💰 Support me!<br>
        Your support helps me maintain and improve the app.
        </p>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <style>
        .bmc-button {
            background-color:#3679AD;
            color:white;
            border:none;
            border-radius:8px;
            padding:10px 20px;
            font-size:16px;
            font-weight:bold;
            cursor:pointer;
            margin-top:5px;
            margin-bottom:18px;
            transition: all 0.3s ease;
        }
        .bmc-button:hover {
            background-color:#003AAB;
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
# Sidebar: Daily token usage
# ---------------------------
TOKEN_FILE = Path("daily_tokens.json")
MAX_TOKENS_PER_DAY = 200_000

def load_daily_tokens():
    today_str = datetime.now(timezone.utc).date().isoformat()
    if TOKEN_FILE.exists():
        data = json.loads(TOKEN_FILE.read_text())
        if data.get("date") == today_str:
            return data.get("tokens", 0)
    return 0

def save_daily_tokens(tokens):
    today_str = datetime.now(timezone.utc).date().isoformat()
    data = {"date": today_str, "tokens": tokens}
    TOKEN_FILE.write_text(json.dumps(data))

with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Daily Token Usage")
    current_tokens = load_daily_tokens()
    used_percentage = min(current_tokens / MAX_TOKENS_PER_DAY, 1.0)
    st.progress(used_percentage, text=f"{used_percentage*100:.1f}%")
    st.markdown(
        "<p style='color:#FCF2D9; font-size:14px;'>ℹ️ Token usage resets every day at 02:00 (Greece local time, UTC+2)</p>",
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
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "input": prompt,
        "max_output_tokens": max_tokens,
        "temperature": temperature
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, data=json.dumps(payload))
        
        if response.status_code == 429:
            return "⚠️ The model usage limit has been reached. Please try again in a few minutes."
        elif response.status_code == 400:
            return "⚠️ The model could not process your request. Try rephrasing or shortening your question."
        elif response.status_code != 200:
            return f"❌ API Error ({response.status_code}): {response.text}"

        result = response.json()

        # Track daily tokens
        limit_tokens = int(response.headers.get("x-ratelimit-limit-tokens", 0))
        remaining_tokens = int(response.headers.get("x-ratelimit-remaining-tokens", 0))
        used_tokens = limit_tokens - remaining_tokens
        current_tokens = load_daily_tokens() + used_tokens
        save_daily_tokens(current_tokens)

        # Parse output
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

        return "⚠️ Unexpected response format from Groq. Please try again."

    except requests.exceptions.RequestException as e:
        return f"❌ Network error while contacting Groq API: {e}"

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
# IMPROVED PDF PROCESSING
# ---------------------------
if st.button("⚙️ Process PDFs"):
    current_files = [f.name for f in uploaded_files]
    if current_files != st.session_state.last_uploaded_files:
        st.session_state.messages = []
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.last_uploaded_files = current_files

    @st.cache_data
    def extract_pdf_texts_with_pages(file_data):
        """Extract text from PDFs page by page with metadata."""
        pdf_pages = []
        for file_name, file_content in file_data:
            reader = PdfReader(BytesIO(file_content))
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                pdf_pages.append({
                    'file': file_name,
                    'page': page_num,
                    'text': text
                })
        return pdf_pages

    file_data = tuple((f.name, f.getvalue()) for f in uploaded_files)
    pdf_pages = extract_pdf_texts_with_pages(file_data)

    def clean_text(text):
        """Clean and normalize text - handle malformed tables."""
        # First, try to detect and fix common table artifacts
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip lines that are just numbers or single words (likely table artifacts)
            if line and not re.match(r'^[\d\s\+\-xd]+$', line) and len(line) > 3:
                cleaned_lines.append(line)
        
        text = ' '.join(cleaned_lines)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove weird characters but keep common punctuation
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        return text.strip()

    def detect_section_header(text):
        """Detect if text contains a section header (all caps, short)."""
        lines = text.split('\n')
        for line in lines[:3]:  # Check first 3 lines
            line = line.strip()
            if line and line.isupper() and len(line) < 100 and len(line) > 5:
                return line
        return None

    def smart_chunk_text(page_text, page_num, file_name, chunk_size=600, overlap=150):
        """
        Create chunks with awareness of section headers and structure.
        Returns list of (chunk_text, metadata) tuples.
        """
        chunks = []
        text = clean_text(page_text)
        
        # Detect section header
        header = detect_section_header(page_text)
        
        # If text is short, keep it as one chunk
        if len(text) < chunk_size:
            metadata = {
                'file': file_name,
                'page': page_num,
                'header': header,
                'chunk_type': 'full_page'
            }
            chunks.append((text, metadata))
            return chunks
        
        # Otherwise, chunk it
        start = 0
        chunk_num = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            
            # If this is the first chunk and we have a header, prepend it
            if chunk_num == 0 and header:
                chunk_text = f"[SECTION: {header}]\n{chunk_text}"
            
            metadata = {
                'file': file_name,
                'page': page_num,
                'header': header,
                'chunk_num': chunk_num,
                'chunk_type': 'partial_page'
            }
            
            chunks.append((chunk_text, metadata))
            start += chunk_size - overlap
            chunk_num += 1
        
        return chunks

    # Process all pages into chunks with metadata
    all_chunks = []
    chunk_metadata = []
    
    for page_data in pdf_pages:
        page_chunks = smart_chunk_text(
            page_data['text'], 
            page_data['page'], 
            page_data['file']
        )
        for chunk_text, metadata in page_chunks:
            all_chunks.append(chunk_text)
            chunk_metadata.append(metadata)

    @st.cache_resource
    def build_faiss_index(chunks):
        # Use a better embedding model - more similar to what Google uses
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index, model, embeddings

    index, model, embeddings = build_faiss_index(all_chunks)

    st.session_state.index = index
    st.session_state.model = model
    st.session_state.embeddings = embeddings
    st.session_state.all_chunks = all_chunks
    st.session_state.chunk_metadata = chunk_metadata
    st.session_state.pdfs_processed = True
    st.session_state.index_ready = True

    st.session_state.pdf_messages = [
        f"✅ Loaded {len(pdf_pages)} pages from {len(set(p['file'] for p in pdf_pages))} PDF(s)",
        f"✅ Created {len(all_chunks)} intelligent chunks with metadata"
    ]

# ---------------------------
# Stop until PDFs are processed
# ---------------------------
if not st.session_state.get("pdfs_processed", False):
    st.stop()

if "pdf_messages" in st.session_state:
    for msg in st.session_state.pdf_messages:
        st.success(msg)

# ---------------------------
# Visual separator
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

    # ---- IMPROVED RAG RETRIEVAL WITH HYBRID SEARCH ----
    
    # Simple BM25 implementation for keyword matching
    def compute_bm25_scores(query, documents, k1=1.5, b=0.75):
        """Compute BM25 scores for documents given a query."""
        # Tokenize
        query_tokens = query.lower().split()
        doc_tokens = [doc.lower().split() for doc in documents]
        
        # Compute document frequencies
        N = len(documents)
        df = Counter()
        for tokens in doc_tokens:
            df.update(set(tokens))
        
        # Compute IDF
        idf = {term: math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1) for term in df}
        
        # Compute scores
        avgdl = sum(len(tokens) for tokens in doc_tokens) / N if N > 0 else 0
        scores = []
        
        for tokens in doc_tokens:
            doc_len = len(tokens)
            score = 0
            term_freqs = Counter(tokens)
            
            for term in query_tokens:
                if term in term_freqs:
                    tf = term_freqs[term]
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (doc_len / avgdl)) if avgdl > 0 else 1
                    score += idf.get(term, 0) * (numerator / denominator)
            
            scores.append(score)
        
        return scores
    
    # 1. Create query variations
    query_lower = query.lower()
    query_variations = [
        query,
        query.upper(),
        query.lower(),
        # Extract key terms
        re.sub(r'\b(tell me about|details about|explain|what are|how do)\b', '', query_lower).strip(),
        # Simplify
        query.replace('?', '').strip(),
    ]
    
    # 2. BM25 keyword search
    bm25_scores = compute_bm25_scores(query, st.session_state.all_chunks)
    # Get top BM25 matches
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:20].tolist()
    
    # 3. Keyword matching for headers
    all_headers = [meta.get('header', '') for meta in st.session_state.chunk_metadata]
    header_matches = []
    for idx, header in enumerate(all_headers):
        if header:
            # Check if query keywords match header
            query_keywords = set(query_lower.split())
            header_keywords = set(header.lower().split())
            if query_keywords & header_keywords:  # If there's any overlap
                header_matches.append(idx)
    
    # 4. Semantic search with multiple queries
    top_k = 15
    semantic_indices = set()
    
    for q in query_variations:
        query_vec = st.session_state.model.encode([q], convert_to_numpy=True)
        distances, indices = st.session_state.index.search(query_vec, top_k)
        semantic_indices.update(indices[0].tolist())
    
    # 5. Combine all three methods: BM25 + Headers + Semantic
    combined_indices = list(set(header_matches + list(semantic_indices) + bm25_top_indices))[:20]
    
    # 6. Get chunks and sort by relevance
    retrieved_chunks = []
    for i, idx in enumerate(combined_indices):
        chunk = st.session_state.all_chunks[idx]
        metadata = st.session_state.chunk_metadata[idx]
        retrieved_chunks.append({
            'text': chunk,
            'metadata': metadata,
            'has_header_match': idx in header_matches,
            'original_index': i  # Track original position
        })
    
    # Sort: header matches first, then by original index
    retrieved_chunks.sort(key=lambda x: (not x['has_header_match'], x['original_index']))
    
    # 6. Format retrieved text with metadata
    formatted_chunks = []
    for chunk_data in retrieved_chunks[:10]:
        meta = chunk_data['metadata']
        header_tag = f"[SECTION: {meta['header']}] " if meta.get('header') else ""
        page_tag = f"(Page {meta['page']}) "
        formatted_chunks.append(f"{header_tag}{page_tag}{chunk_data['text']}")
    
    retrieved_text = "\n\n---\n\n".join(formatted_chunks)

    # 7. Show debug info
    with st.expander("🔍 Retrieved Context (for debugging)", expanded=False):
        st.write("**BM25 keyword matches found:**", len(bm25_top_indices))
        st.write("**Header matches found:**", len(header_matches))
        st.write("**Semantic matches found:**", len(semantic_indices))
        st.write("**Combined unique chunks:**", len(combined_indices))
        st.markdown("---")
        st.text(retrieved_text[:3000] + "..." if len(retrieved_text) > 3000 else retrieved_text)

    # ---- Prompt construction ----
    recent_history = "\n".join(
        [f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages[-3:]]
    )

    prompt = f"""
You are a board game rules expert.

Here are the most relevant excerpts from the rulebook (with page numbers and section headers):

{retrieved_text}

Recent conversation (for reference only — ignore if unrelated):
{recent_history}

User's question: {query}

Instructions:
- Answer clearly and concisely using ONLY information from the rulebook excerpts above.
- Pay special attention to [SECTION: ...] tags - these indicate section headers.
- If you see a section that directly matches the user's query, use that information.
- Include page numbers when referencing specific rules.
- Keep your answer under 1200 tokens.
- If the answer isn't found in the excerpts, reply: "That information cannot be found in the provided PDFs."
"""

    # ---- Generate answer ----
    with st.chat_message("assistant"):
        with st.spinner("Meditating..."):
            answer = groq_generate(prompt, max_tokens=3000, temperature=0.3)
            st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})