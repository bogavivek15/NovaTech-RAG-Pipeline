import os
import re
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import chromadb
from typing import List

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY:
    GROQ_API_KEY = GROQ_API_KEY.strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not GROQ_API_KEY:
    print("⚠️  WARNING: GROQ_API_KEY not set! Add it to .env file")
    client = None
else:
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )
    print(f"✅ Groq client initialized with model: {GROQ_MODEL}")

chroma_client = chromadb.Client()

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]

class UploadResponse(BaseModel):
    message: str
    files_processed: int
    chunks_added: int

collection = None
documents_loaded = False


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


def initialize_collection():
    """Initialize ChromaDB collection."""
    global collection
    if collection is None:
        collection = chroma_client.create_collection(name='company_docs')
        print("✅ ChromaDB collection created")
    return collection


def find_data_folder():
    """Find the _data folder."""
    current_folder = os.path.dirname(__file__)
    data_folder = os.path.join(current_folder, "_data")
    
    if os.path.isdir(data_folder):
        return data_folder
    
    raise FileNotFoundError("Cannot find _data folder with documents!")


def load_documents():
    """Load all text files from _data folder and store them in ChromaDB."""
    global collection, documents_loaded
    
    if documents_loaded:
        return
    
    print("📂 Loading documents from _data folder...")
    initialize_collection()
    
    try:
        data_folder = find_data_folder()
    except FileNotFoundError:
        print("⚠️  No _data folder found. Skipping initial document load.")
        documents_loaded = True
        return
    
    all_chunks = []
    
    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            
            chunks = chunk_document(text, filename)
            all_chunks.extend(chunks)
            print(f"   📄 {filename}: {len(chunks)} chunks")
    
    documents = []
    ids = []
    metadatas = []
    
    for i, chunk in enumerate(all_chunks):
        documents.append(chunk["text"])
        ids.append(f"chunk_{i}")
        metadatas.append({"source": chunk["source"]})
    
    collection.add(documents=documents, ids=ids, metadatas=metadatas)
    documents_loaded = True
    print(f"✅ Loaded {len(documents)} chunks into ChromaDB\n")


def process_uploaded_file(content: bytes, filename: str):
    """Process an uploaded file and add it to ChromaDB."""
    global collection
    
    if collection is None:
        initialize_collection()
    
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1")
    
    chunks = chunk_document(text, filename)
    
    try:
        existing_count = collection.count()
    except:
        existing_count = 0
    
    documents = [chunk["text"] for chunk in chunks]
    ids = [f"chunk_{existing_count + i}" for i in range(len(chunks))]
    metadatas = [{"source": chunk["source"]} for chunk in chunks]
    
    collection.add(documents=documents, ids=ids, metadatas=metadatas)
    return len(chunks)


def retrieve(question, n_results=3):
    """Retrieve relevant chunks from ChromaDB."""
    if collection is None:
        raise HTTPException(status_code=503, detail="Documents not loaded yet. Please try again.")
    
    results = collection.query(query_texts=[question], n_results=n_results)
    return results["documents"][0], results["metadatas"][0]


def ask_rag(question, n_results=3):
    """Ask a question using RAG pipeline."""
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
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.2
        )
        answer = response.choices[0].message.content
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
        answer = re.sub(r'<think>.*', '', answer, flags=re.DOTALL)
        answer = answer.strip()
    except Exception as error:
        print(f"❌ Groq API error: {error}")
        raise HTTPException(status_code=502, detail=f"Groq API error: {error}")
    
    unique_sources = list(set([s["source"] for s in sources]))
    return answer, unique_sources


app = FastAPI(
    title="NovaTech RAG Chatbot",
    version="1.0",
    description="RAG-powered chatbot using Groq and ChromaDB"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.on_event("startup")
def startup_event():
    print("🚀 Starting NovaTech RAG Chatbot...")
    print(f"   Model: {GROQ_MODEL}")
    load_documents()
    print("✅ RAG pipeline ready!")


@app.get("/")
def home():
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    index_path = os.path.join(static_dir, "index.html")
    
    if os.path.exists(index_path):
        return FileResponse(index_path)
    
    return {
        "message": "Welcome to NovaTech RAG Chatbot!",
        "model": GROQ_MODEL,
        "endpoints": {
            "chat": "POST /chat",
            "upload": "POST /upload",
            "health": "GET /health"
        }
    }


@app.get("/health")
def health_check():
    doc_count = 0
    if collection:
        try:
            doc_count = collection.count()
        except:
            pass
    
    return {
        "status": "running",
        "groq_api_key_configured": bool(GROQ_API_KEY),
        "model": GROQ_MODEL,
        "collection_initialized": collection is not None,
        "document_chunks": doc_count,
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
            print(f"⚠️  Skipping {file.filename} - only .txt and .md files supported")
            continue
        
        try:
            content = await file.read()
            chunks_added = process_uploaded_file(content, file.filename)
            files_processed += 1
            total_chunks += chunks_added
            print(f"✅ Processed {file.filename}: {chunks_added} chunks added")
        except Exception as e:
            print(f"❌ Error processing {file.filename}: {str(e)}")
            continue
    
    if files_processed == 0:
        raise HTTPException(status_code=400, detail="No valid files were processed.")
    
    return UploadResponse(
        message=f"Successfully processed {files_processed} file(s)",
        files_processed=files_processed,
        chunks_added=total_chunks
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    print(f"\n{'═' * 60}")
    print(f"❓ Question: {request.question}")
    print(f"{'─' * 60}")
    
    try:
        answer, sources = ask_rag(request.question, n_results=3)
        print(f"📚 Sources: {sources}")
        print(f"🤖 Answer: {answer[:100]}...")
        print(f"{'═' * 60}\n")
    except HTTPException:
        raise
    except Exception as error:
        print(f"❌ Unexpected error: {error}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {error}")
    
    return ChatResponse(answer=answer, sources=sources)
