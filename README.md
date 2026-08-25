# NovaTech RAG Chatbot 🤖

A powerful AI chatbot that answers questions about company documents using RAG (Retrieval Augmented Generation) with Groq's LLM and ChromaDB.

## Features

- 🤖 AI-powered question answering using Groq API
- 📚 Semantic search through company documents
- 📤 Upload documents via web interface
- 💬 Clean HTML/CSS/JS frontend
- 🔌 RESTful API with FastAPI
- 🐳 Docker support

## Quick Start

### 1. Get Groq API Key (FREE)

Sign up at [https://console.groq.com/keys](https://console.groq.com/keys)

### 2. Configure Environment

Edit `.env`:
```env
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=qwen/qwen3.6-27b
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Or use `start.bat` on Windows

### 5. Open Browser

- Chat Interface: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## Project Structure

```
novatech/
├── main.py                 # FastAPI backend
├── static/                 # Frontend
│   ├── index.html
│   ├── style.css
│   └── script.js
├── _data/                  # Company documents
├── requirements.txt
├── Dockerfile
├── .env                    # Configuration
└── README.md
```

## API Endpoints

### POST /chat
Ask a question
```json
{
  "question": "What is the work from home policy?"
}
```

### POST /upload
Upload documents (multipart/form-data)

### GET /health
Check system status

## Docker

```bash
docker build -t novatech-rag .
docker run -p 8000:8000 --env-file .env novatech-rag
```

## License

MIT
