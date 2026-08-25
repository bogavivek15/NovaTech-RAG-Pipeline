import os
import re
import math
from collections import Counter
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from typing import List

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY:
    GROQ_API_KEY = GROQ_API_KEY.strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not set! Add it to .env file")
    client = None
else:
    client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    print(f"Groq client initialized with model: {GROQ_MODEL}")

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]

class UploadResponse(BaseModel):
    message: str
    files_processed: int
    chunks_added: int

# Lightweight in-memory lexical vector store.
# This replaces Chroma's bundled ONNX embedding model so the app fits Render Free.
chunks_store = []
documents_loaded = False
vocabulary = {}
idf = {}
tf_vectors = []


def chunk_document(text, source_name):
    """Split document into chunks by paragraphs."""
    paragraphs = text.strip().split("\n\n")
    chunks = []
    for para in paragraphs:
        para = para.strip()
        if len(para) < 50 or para.startswith("====="):
            continue
        chunks.append({"text": para, "source": source_name})
    return chunks


def tokenize(text):
    """Small, dependency-free tokenizer for lightweight retrieval."""
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def rebuild_index():
    """Build a TF-IDF index entirely in memory."""
    global vocabulary, idf, tf_vectors
    tokenized = [tokenize(item["text"]) for item in chunks_store]
    vocabulary = {term: i for i, term in enumerate(sorted({t for doc in tokenized for t in doc}))}
    doc_count = len(tokenized)
    if not doc_count:
        vocabulary, idf, tf_vectors = {}, {}, []
        return

    df = Counter()
    for tokens in tokenized:
        df.update(set(tokens))
    idf = {term: math.log((1 + doc_count) / (1 + freq)) + 1.0 for term, freq in df.items()}

    tf_vectors = []
    for tokens in tokenized:
        counts = Counter(tokens)
        total = max(len(tokens), 1)
        vector = {}
        norm_sq = 0.0
        for term, count in counts.items():
            if term in idf:
                weight = (count / total) * idf[term]
                vector[term] = weight
                norm_sq += weight * weight
        norm = math.sqrt(norm_sq) or 1.0
        tf_vectors.append({term: weight / norm for term, weight in vector.items()})


def initialize_collection():
    """Compatibility helper for the existing application flow."""
    return True


def find_data_folder():
    current_folder = os.path.dirname(__file__)
    data_folder = os.path.join(current_folder, "_data")
    if os.path.isdir(data_folder):
        return data_folder
    raise FileNotFoundError("Cannot find _data folder with documents!")


def load_documents():
    """Load text files and build the lightweight retrieval index."""
    global documents_loaded
    if documents_loaded:
        return

    print("Loading documents from _data folder...")
    initialize_collection()
    try:
        data_folder = find_data_folder()
    except FileNotFoundError:
        print("WARNING: No _data folder found. Skipping initial document load.")
        documents_loaded = True
        return

    chunks_store.clear()
    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                chunks = chunk_document(f.read(), filename)
            chunks_store.extend(chunks)
            print(f"   {filename}: {len(chunks)} chunks")

    rebuild_index()
    documents_loaded = True
    print(f"Loaded {len(chunks_store)} chunks into lightweight retrieval index")


def process_uploaded_file(content: bytes, filename: str):
    """Process an uploaded file and add it to the retrieval index."""
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1")

    chunks = chunk_document(text, filename)
    chunks_store.extend(chunks)
    rebuild_index()
    return len(chunks)


def retrieve(question, n_results=3):
    """Retrieve the most relevant chunks using cosine similarity over TF-IDF vectors."""
    if not chunks_store:
        raise HTTPException(status_code=503, detail="Documents not loaded yet. Please try again.")

    query_tokens = tokenize(question)
    query_counts = Counter(query_tokens)
    total = max(len(query_tokens), 1)
    query_vector = {}
    for term, count in query_counts.items():
        if term in idf:
            query_vector[term] = (count / total) * idf[term]
    query_norm = math.sqrt(sum(value * value for value in query_vector.values())) or 1.0
    query_vector = {term: value / query_norm for term, value in query_vector.items()}

    scored = []
    for index, vector in enumerate(tf_vectors):
        score = sum(query_vector.get(term, 0.0) * weight for term, weight in vector.items())
        scored.append((score, index))

    scored.sort(reverse=True)
    selected = [item for item in scored[:n_results] if item[0] > 0]
    if not selected:
        # Return a small fallback set so the LLM can still respond appropriately.
        selected = scored[:n_results]

    documents = [chunks_store[index]["text"] for _, index in selected]
    metadata = [{"source": chunks_store[index]["source"]} for _, index in selected]
    return documents, metadata


def ask_rag(question, n_results=3):
    """Ask a question using the RAG pipeline."""
    if client is None:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not set. Add it to .env file and restart.")

    chunks, sources = retrieve(question, n_results)
    if not chunks:
        return "I don't have enough information to answer this question.", []

    context = "\n\n".join(chunks)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that answers questions based only on the provided context. "
                "If the context does not contain enough information, say 'I don't have enough information.' "
                "Be concise and professional. For general questions, use your knowledge. "
                "IMPORTANT: Provide only the final answer without reasoning or tags."
            )
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.2
        )
        answer = response.choices[0].message.content
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
        answer = re.sub(r'<think>.*', '', answer, flags=re.DOTALL).strip()
    except Exception as error:
        print(f"Groq API error: {error}")
        raise HTTPException(status_code=502, detail=f"Groq API error: {error}")

    unique_sources = list(dict.fromkeys(s["source"] for s in sources))
    return answer, unique_sources


app = FastAPI(
    title="NovaTech RAG Chatbot",
    version="1.0",
    description="RAG-powered chatbot using Groq and lightweight TF-IDF retrieval"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.on_event("startup")
def startup_event():
    print("Starting NovaTech RAG Chatbot...")
    print(f"   Model: {GROQ_MODEL}")
    load_documents()
    print("RAG pipeline ready!")


@app.get("/")
def home():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": "Welcome to NovaTech RAG Chatbot!",
        "model": GROQ_MODEL,
        "endpoints": {"chat": "POST /chat", "upload": "POST /upload", "health": "GET /health"}
    }


@app.get("/health")
def health_check():
    return {
        "status": "running",
        "groq_api_key_configured": bool(GROQ_API_KEY),
        "model": GROQ_MODEL,
        "retriever": "tfidf",
        "document_chunks": len(chunks_store),
        "documents_loaded": documents_loaded
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    files_processed = 0
    total_chunks = 0
    for file in files:
        if not file.filename.endswith((".txt", ".md")):
            print(f"Skipping {file.filename} - only .txt and .md files supported")
            continue
        try:
            content = await file.read()
            chunks_added = process_uploaded_file(content, file.filename)
            files_processed += 1
            total_chunks += chunks_added
            print(f"Processed {file.filename}: {chunks_added} chunks added")
        except Exception as e:
            print(f"Error processing {file.filename}: {str(e)}")

    if files_processed == 0:
        raise HTTPException(status_code=400, detail="No valid files were processed.")

    return UploadResponse(
        message=f"Successfully processed {files_processed} file(s)",
        files_processed=files_processed,
        chunks_added=total_chunks
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    print(f"\n{'=' * 60}")
    print(f"Question: {request.question}")
    print(f"{'-' * 60}")
    try:
        answer, sources = ask_rag(request.question, n_results=3)
        print(f"Sources: {sources}")
        print(f"Answer: {answer[:100]}...")
    except HTTPException:
        raise
    except Exception as error:
        print(f"Unexpected error: {error}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {error}")

    return ChatResponse(answer=answer, sources=sources)
