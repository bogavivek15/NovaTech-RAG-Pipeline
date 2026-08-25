# NovaTech RAG Chatbot 🤖

A powerful AI chatbot that answers questions about company documents using RAG (Retrieval Augmented Generation) with Groq's LLM and ChromaDB.

## Features ✨

- 🔍 **Semantic Search** - ChromaDB vector database for intelligent document retrieval
- 🤖 **AI-Powered** - Uses Groq's llama-3.3-70b-versatile model
- 📄 **Document Upload** - Upload your own .txt or .md files
- 💬 **Modern UI** - Clean, responsive chat interface
- 🚀 **Fast API** - Built with FastAPI for high performance

## Tech Stack

- **Backend**: FastAPI, Python
- **AI**: Groq API (OpenAI-compatible)
- **Vector DB**: ChromaDB
- **Frontend**: Vanilla JS, HTML, CSS

## Quick Start

### Prerequisites
- Python 3.11+
- Groq API key (get it free at [console.groq.com](https://console.groq.com))

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/bogavivek15/NovaTech-RAG-Pipeline.git
cd NovaTech-RAG-Pipeline
```

2. Create a `.env` file:
```
GROQ_API_KEY=your_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
uvicorn main:app --reload
```

5. Open http://localhost:8000

### Docker Setup

```bash
docker build -t novatech-rag .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key_here novatech-rag
```

## Free Deployment Options 🚀

### Option 1: Render (Recommended)

1. Fork/clone this repository
2. Go to [render.com](https://render.com)
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Render will auto-detect the `render.yaml` configuration
6. Add your `GROQ_API_KEY` in the environment variables
7. Click "Create Web Service"

### Option 2: Railway

1. Go to [railway.app](https://railway.app)
2. Click "New Project" → "Deploy from GitHub repo"
3. Select this repository
4. Add environment variable: `GROQ_API_KEY`
5. Railway will automatically use the Dockerfile
6. Deploy!

### Option 3: Hugging Face Spaces

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Select "Docker" as the SDK
4. Upload your files or connect via Git
5. Add `GROQ_API_KEY` in Settings → Repository secrets
6. Your app will be live at `username-spacename.hf.space`

## API Endpoints

- `GET /` - Web interface
- `POST /chat` - Send a question
- `POST /upload` - Upload documents
- `GET /health` - Check system status

## Project Structure

```
.
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
├── render.yaml            # Render deployment config
├── static/                # Frontend files
│   ├── index.html
│   ├── script.js
│   └── style.css
└── _data/                 # Default knowledge base
    ├── company_hr_policy.txt
    ├── engineering_standards.txt
    ├── onboarding_guide.txt
    ├── product_knowledge_base.txt
    └── security_policy.txt
```

## Environment Variables

- `GROQ_API_KEY` - Your Groq API key (required)
- `GROQ_MODEL` - Model to use (default: llama-3.3-70b-versatile)

## Contributing

Pull requests are welcome! For major changes, please open an issue first.

## License

MIT

## Author

**Vivek Boga**
- GitHub: [@bogavivek15](https://github.com/bogavivek15)
