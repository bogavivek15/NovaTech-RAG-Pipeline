import os
import re
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from fastembed import TextEmbedding
from typing import List

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY:
    GROQ_API_KEY = GROQ_API_KEY.strip()

GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
TOP_K = 3
MIN_SIMILARITY = 0.30
EMBED_BATCH_SIZE = 1

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not set!")
    client = None
else:
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )
    print(f"Groq client initialized with model: {GROQ_MODEL}")

# BGE-small is a lightweight semantic embedding model.
# FastEmbed runs it with ONNX instead of loading a large PyTorch stack.
# One thread and batch size 1 minimize peak RAM on Render's 512 MB instance.
print(f"Loading embedding model: {EMBEDDING_MODEL}")
embedding_model = TextEmbedding(
    model_name=EMBEDDING_MODEL,
    threads=1
)
print("Embedding model ready!")


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


class UploadResponse(BaseModel):
    message: str
    files_processed: int
    chunks_added: int


chunks_store = []
document_embeddings = np.empty((0, 384), dtype=np.float32)
documents_loaded = False


def chunk_document(text, source_name):
    paragraphs = re.split(r"\n\s*\n+", text.strip())
    chunks = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if len(paragraph) < 50 or paragraph.startswith("====="):
            continue
        chunks.append({"text": paragraph, "source": source_name})

    return chunks


def rebuild_embeddings():
    """Create semantic embeddings one chunk at a time to minimize RAM."""
    global document_embeddings

    if not chunks_store:
        document_embeddings = np.empty((0, 384), dtype=np.float32)
        return

    total = len(chunks_store)
    print(f"Creating semantic embeddings for {total} chunks...")

    vectors = np.empty((total, 384), dtype=np.float32)

    for index, item in enumerate(chunks_store):
        text = "passage: " + item["text"]
        vector = next(embedding_model.passage_embed([text], batch_size=1))
        vector = np.asarray(vector, dtype=np.float32)
        vector /= max(np.linalg.norm(vector), 1e-12)
        vectors[index] = vector

        if (index + 1) % 10 == 0 or index + 1 == total:
            print(f"   Embedded {index + 1}/{total} chunks")

    document_embeddings = vectors

    print(
        f"Semantic index ready: {document_embeddings.shape[0]} vectors x "
        f"{document_embeddings.shape[1]} dimensions"
    )


def find_data_folder():
    data_folder = os.path.join(os.path.dirname(__file__), "_data")
    if os.path.isdir(data_folder):
        return data_folder
    raise FileNotFoundError("Cannot find _data folder with documents!")


def load_documents():
    global documents_loaded

    if documents_loaded:
        return

    print("Loading documents from _data folder...")

    try:
        data_folder = find_data_folder()
    except FileNotFoundError:
        print("WARNING: No _data folder found.")
        documents_loaded = True
        return

    chunks_store.clear()

    for filename in os.listdir(data_folder):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(data_folder, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            chunks_store.extend(chunk_document(file.read(), filename))

        print(f"   {filename}: {len(chunk_document(open(filepath, encoding='utf-8').read(), filename))} chunks")

    rebuild_embeddings()
    documents_loaded = True
    print(f"Loaded {len(chunks_store)} chunks into semantic retrieval index")


def process_uploaded_file(content: bytes, filename: str):
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1")

    chunks_store.extend(chunk_document(text, filename))
    rebuild_embeddings()
    return len(chunk_document(text, filename))


def retrieve(question, n_results=TOP_K):
    if not chunks_store or document_embeddings.size == 0:
        raise HTTPException(status_code=503, detail="Documents are not loaded yet. Please try again.")

    query_vector = next(embedding_model.query_embed(["query: " + question]))
    query_vector = np.asarray(query_vector, dtype=np.float32)
    query_vector /= max(np.linalg.norm(query_vector), 1e-12)

    scores = np.dot(document_embeddings, query_vector)
    ranked_indexes = np.argsort(scores)[::-1]

    selected = []
    for index in ranked_indexes[:n_results]:
        score = float(scores[index])
        if score >= MIN_SIMILARITY:
            selected.append((score, int(index)))

    if not selected:
        return [], []

    documents = [chunks_store[index]["text"] for _, index in selected]
    sources = [{"source": chunks_store[index]["source"]} for _, index in selected]

    print("Retrieval scores:", [round(score, 3) for score, _ in selected])
    print("Retrieved sources:", list(dict.fromkeys(s["source"] for s in sources)))
    return documents, sources


def ask_rag(question, n_results=TOP_K):
    if client is None:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not set. Add it to Render environment variables.")

    chunks, sources = retrieve(question, n_results)
    if not chunks:
        return "I do not have enough information in the NovaTech knowledge base.", []

    context = "\n\n".join(chunks)
    messages = [
        {"role": "system", "content": "You are the NovaTech company assistant. Use the provided context to answer NovaTech-specific questions. Do not invent company policies, procedures, employee records, or product details. If the context does not contain enough information, say exactly: 'I do not have enough information in the NovaTech knowledge base.' For ordinary general questions, answer normally when the question is clearly not asking about NovaTech. Be concise and professional. Provide only the final answer without reasoning or tags."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.7,
            top_p=0.8,
            presence_penalty=1.5,
            reasoning_effort="none",
            max_tokens=1024
        )
        answer = response.choices[0].message.content or "I do not have enough information in the NovaTech knowledge base."
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)
        answer = re.sub(r"<think>.*", "", answer, flags=re.DOTALL).strip()
    except Exception as error:
        print(f"Groq API error: {error}")
        raise HTTPException(status_code=502, detail=f"Groq API error: {error}")

    return answer, list(dict.fromkeys(s["source"] for s in sources))


app = FastAPI(title="NovaTech RAG Chatbot", version="2.0", description="RAG chatbot using lightweight semantic embeddings and Groq")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.on_event("startup")
def startup_event():
    print("Starting NovaTech RAG Chatbot...")
    print(f"   Answer model: {GROQ_MODEL}")
    print(f"   Embedding model: {EMBEDDING_MODEL}")
    load_documents()
    print("RAG pipeline ready!")


@app.get("/")
def home():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Welcome to NovaTech RAG Chatbot!", "model": GROQ_MODEL, "embedding_model": EMBEDDING_MODEL, "endpoints": {"chat": "POST /chat", "upload": "POST /upload", "health": "GET /health"}}


@app.get("/health")
def health_check():
    dimensions = int(document_embeddings.shape[1]) if document_embeddings.size else 0
    return {"status": "running", "groq_api_key_configured": bool(GROQ_API_KEY), "model": GROQ_MODEL, "retriever": "semantic", "embedding_model": EMBEDDING_MODEL, "embedding_dimensions": dimensions, "document_chunks": len(chunks_store), "documents_loaded": documents_loaded}


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
        except Exception as error:
            print(f"Error processing {file.filename}: {error}")

    if files_processed == 0:
        raise HTTPException(status_code=400, detail="No valid files were processed.")

    return UploadResponse(message=f"Successfully processed {files_processed} file(s)", files_processed=files_processed, chunks_added=total_chunks)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    print("\n" + "=" * 60)
    print(f"Question: {question}")
    print("-" * 60)

    try:
        answer, sources = ask_rag(question, n_results=TOP_K)
        print(f"Sources: {sources}")
        print(f"Answer: {answer[:100]}...")
    except HTTPException:
        raise
    except Exception as error:
        print(f"Unexpected error: {error}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {error}")

    return ChatResponse(answer=answer, sources=sources)
