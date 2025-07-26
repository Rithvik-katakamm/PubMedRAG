# 🧠 PubMed RAG - Medical Research Assistant

## 🩺 Business Problem

Clinicians, researchers, and data scientists often need up-to-date insights from **PubMed** but end up manually reading dozens of abstracts or forming complex search queries. This slows down literature review and clinical decision-making.

---

## 🎯 Objective

Provide a **chat-style assistant** that:

1. Pulls the most relevant PubMed abstracts for a given topic in real-time.
2. Leverages **Retrieval-Augmented Generation (RAG)** so GPT-4o mini can answer follow-up questions grounded in those abstracts.
3. Available in both **terminal** and **modern web interface**.

---

## ⚙️ How It Works (High-Level)

User ⇄ Chat Interface (Terminal or Web)
        ->
   PubMed fetch (Entrez)
        ->
   Text chunking + BioBERT embedding
        ->
   ChromaDB vector storage
        ->
      GPT-4o answer

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Rithvik-katakamm/PubMedRAG.git
cd PubMedRAG

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

### 2. Set Your API Keys

Create environment variables or a `.env` file:

```env
OPENAI_API_KEY=sk-...
ENTREZ_EMAIL=your_email@example.com
```

---

### 3. Choose Your Interface

#### 🌐 **Web Interface** (Recommended)
Beautiful Gemini-inspired dark theme with modern UI:

```bash
./run_web.sh
# Then open: http://localhost:8000
```

**Features:**
- Dark theme with blue accents (like Gemini)
- Real-time text streaming
- Collapsible session sidebar
- Floating action buttons for chunks/metrics
- Elegant popups for viewing retrieved chunks
- Session management and history

#### 💻 **Terminal Interface**
Rich console-based interface:

```bash
./run.sh
# or
python src/app.py
```

**Features:**
- Beautiful terminal UI with Rich library
- Session persistence
- All slash commands (`/chunks`, `/clear`, `/help`)
- Progress indicators

---

## 🎨 Web Interface Preview

The web interface features:
- **Centered chat** like Gemini
- **Collapsible sidebar** for session management  
- **Topic indicator** in header
- **Floating action buttons** for:
  - 📄 View retrieved chunks
  - 📊 Data loading metrics
  - 🗑️ Clear data
  - ➕ New session
- **Elegant modals** for chunks and metrics
- **Real-time streaming** responses

---

## 🛠️ Tech Stack

### **Core**
- Python 3.8+
- OpenAI GPT-4o mini
- ChromaDB (vector database)
- BioBERT (medical embeddings)
- NCBI Entrez (PubMed API)

### **Web Version**
- FastAPI (backend)
- WebSockets (real-time communication)
- Modern HTML/CSS/JavaScript
- Dark theme with responsive design

### **Terminal Version**
- Rich (beautiful console output)
- Click (command-line interface)

---

## 📁 Project Structure

```
rag_pubmed/
├── src/
│   ├── web_app.py          # FastAPI web application
│   ├── app.py              # Terminal application  
│   ├── static/             # Web UI assets
│   │   ├── style.css       # Gemini-inspired styling
│   │   └── script.js       # Frontend JavaScript
│   ├── core/               # RAG engine
│   │   ├── rag_engine.py   # Main orchestrator
│   │   ├── vector_db.py    # ChromaDB manager
│   │   ├── retrieval.py    # PubMed fetching
│   │   ├── chunking.py     # Text processing
│   │   └── embeddings.py   # BioBERT embeddings
│   ├── chat/               # Session management
│   └── ui/                 # Terminal UI
├── run_web.sh              # Web app launcher
├── run.sh                  # Terminal app launcher
└── requirements.txt        # Dependencies
```

---

## 🔧 Configuration

### **Environment Variables**
- `OPENAI_API_KEY` - Your OpenAI API key
- `ENTREZ_EMAIL` - Email for NCBI Entrez API

### **Performance Tuning**
The system is optimized for M2 Macs with:
- Max 5000 vectors per topic (sub-100ms retrieval)
- HNSW indexing for fast similarity search
- Automatic FIFO cleanup

---

## 💡 Usage Examples

### Starting a New Session
1. **Web**: Click ➕ button → Enter email and topic
2. **Terminal**: Run app → Enter email and topic

### Asking Questions
- "What are the latest treatments for diabetes?"
- "How effective is immunotherapy for lung cancer?"
- "What are the side effects of metformin?"

### Viewing Retrieved Sources
- **Web**: Click 📄 button for elegant popup
- **Terminal**: Type `/chunks` or `c`

### Session Management
- **Web**: Use collapsible sidebar
- **Terminal**: Use `/sessions` command

---

## 🧼 Notes

* Sessions are persistent across restarts
* Vector data is stored in `./data/chroma_db/`
* Each topic maintains its own collection
* FIFO cleanup prevents memory issues
* Works offline once data is loaded

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

MIT License - see LICENSE file for details.

---

**Switch to Ollama**: You can replace OpenAI with Ollama for fully local operation! 


