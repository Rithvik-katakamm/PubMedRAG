# 🧠 PubMed RAG (No-DB Edition)

## 🩺 Business Problem

Clinicians, researchers, and data scientists often need up-to-date insights from **PubMed** but end up manually reading dozens of abstracts or forming complex search queries. This slows down literature review and clinical decision-making.

---

## 🎯 Objective

Provide a **chat-style assistant** that:

1. Pulls the most relevant PubMed abstracts for a given topic in real-time.
2. Leverages **Retrieval-Augmented Generation (RAG)** so GPT-4o can answer follow-up questions grounded in those abstracts.
3. Runs **locally with no database dependency** — all vector data is kept only for the life of the chat and deleted on exit.

---

## ⚙️ How It Works (High-Level)

User ⇄ CLI chat loop
        ↓
   PubMed fetch (Entrez)
        ↓
   Text chunking + Biobert embedding
        ↓
   Chroma vector DB (ephemeral per chat)
        ↓
      GPT-4o answer

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Rithvik-katakamm/PubMedRAG.git
cd PubMedRAG

# Create and activate virtual environment
python -m venv rag_env
source rag_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### 2. Set Your Secrets

Create a file called `env_variables.env` (or export these variables directly):

```env
OPENAI_API_KEY=sk-...
ENTREZ_EMAIL=your_email@example.com
```

---

### 3. Run the Chat

```bash
python PubMedRag_noDB.py
```

---

### 4. Sample Session

```text
NCBI Entrez email: your_email@example.com  
PubMed search topic: dialysis  
Abstracts indexed.

You: how is home dialysis different from in-center dialysis?
Assistant: ...
```

---

## 🧼 Notes

* This version avoids persistent DB writes — it's ideal for demoing or testing ephemeral RAG workflows.
* Vector DB (Chroma) is memory-backed and wiped after each session.

---

## 🧠 Tech Stack

* Python
* OpenAI GPT-4o
* HuggingFace Transformers (BioBERT)
* ChromaDB (in-memory)
* Entrez (NCBI API)



