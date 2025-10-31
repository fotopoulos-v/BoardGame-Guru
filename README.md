<p align="center">
  <img src="assets/images/guru_logo.png" alt="BoardGame Guru Logo" width="110"/>
</p>

<h1 align="center">BoardGame Guru</h1>


**Your AI rules master!**  
Upload your board game rulebooks and ask questions in plain English — BoardGame Guru reads your PDFs and explains the rules so you can focus on playing.  

🧠 Powered by advanced language models and embeddings for fast, accurate answers.

---

## 🚀 Try It Live

👉 **[Launch BoardGame Guru on Streamlit →](https://boardgame-guru.streamlit.app/)**

---

## 💡 Features

✅ Upload multiple board game rulebooks (PDFs)  
✅ Ask questions in natural English — no keywords needed  
✅ AI retrieves relevant passages using semantic search (FAISS + embeddings)  
✅ Clean, modern interface built with Streamlit  
✅ Free to use — no login required  

---

## 🧩 How It Works

1. **Upload** one or more PDF rulebooks.  
2. **BoardGame Guru** extracts text and splits it into chunks.  
3. Each chunk is **embedded** using the `all-MiniLM-L6-v2` model.  
4. A **FAISS vector database** stores these embeddings for efficient similarity search.  
5. When you ask a question, the top matching passages are retrieved and sent to the **Groq LLM (`openai/gpt-oss-120b`)**.  
6. The model answers **only** using the content from your uploaded PDFs.

---

## 🧠 Models Used

| Purpose | Model | Library |
|----------|--------|----------|
| Embeddings | `all-MiniLM-L6-v2` | `sentence-transformers` |
| Chat / Reasoning | `openai/gpt-oss-120b` | via `groq` API |
| Vector Indexing | FAISS | `faiss-cpu` |

---

## 🛠️ Tech Stack

| Component | Library |
|------------|----------|
| Web app | [Streamlit](https://streamlit.io/) |
| PDF reading | [PyPDF](https://pypi.org/project/pypdf/) |
| Embeddings | [SentenceTransformers](https://www.sbert.net/) |
| Vector search | [FAISS](https://github.com/facebookresearch/faiss) |
| Data handling | NumPy |
| HTTP requests | Requests |


