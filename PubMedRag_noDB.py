"""
pubmed_rag_chat.py  –  CLI PubMed RAG chat (no Postgres)

Features
--------
1. Fetch PubMed abstracts for a user‑supplied topic
2. Split into chunks, embed with BioBERT
3. Store/retrieve in a temporary Chroma collection
4. Chat with OpenAI (GPT‑4‑turbo default) using Retrieval‑Augmented Generation
5. Clean up vector store on exit

Requirements
------------
pip install biopython chromadb sentence-transformers langchain openai python-dotenv
"""

import os, uuid, logging
from typing import List, Dict, Any, Optional, Union

from dotenv import load_dotenv
from Bio import Entrez
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from openai import OpenAI

# ──────────────────────────────  configuration  ──────────────────────────────

load_dotenv("env_variables.env")          # expect OPENAI_API_KEY, ENTrez email, ...

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────  utilities  ──────────────────────────────


def fetch_pubmed_abstracts(
    email: str,
    topic: str,
    db: str = "pubmed",
    retmax: int = 10,
) -> str:
    """Fetch abstracts for *topic* and return them as one long string."""
    Entrez.email = email
    handle = Entrez.esearch(db=db, term=topic, retmax=retmax)
    record = Entrez.read(handle)
    id_list = record["IdList"]
    logger.info(f"Found {len(id_list)} articles for '{topic}'")

    handle = Entrez.efetch(db=db, id=id_list, rettype="abstract", retmode="text")
    return handle.read()


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 50,
    separators: Optional[List[str]] = None,
) -> List[str]:
    """Split *text* into overlapping chunks."""
    if separators is None:
        separators = ["\n\n", "\n", ".", "!", "?", " ", ""]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators
    )
    docs = splitter.create_documents([text])
    chunks = [d.page_content for d in docs]
    logger.info(f"Chunks created: {len(chunks)}")
    return chunks


def embed_chunks(
    model: SentenceTransformer,
    chunks: List[str],
) -> List[List[float]]:
    """Return list of embeddings (lists of floats)."""
    emb = model.encode(chunks, show_progress_bar=True)
    logger.info(f"Generated {len(emb)} embeddings, size {emb.shape[1]}")
    return emb.tolist()


# ──────────────────────────────  Chroma manager  ──────────────────────────────


class ChromaDBManager:
    """Owns one collection (unique per chat)."""

    def __init__(self, collection_name: str, db_path: str = "./chroma_db"):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE,
        )

    def collection(self):
        return self.client.get_or_create_collection(name=self.collection_name)

    def add_chunks(self, chunks: List[str], embeddings: List[List[float]]):
        ids = [str(uuid.uuid4()) for _ in chunks]
        self.collection().add(ids=ids, documents=chunks, embeddings=embeddings)

    def query(self, query_text: str, model: SentenceTransformer, k: int = 10):
        q_emb = model.encode([query_text]).tolist()
        return self.collection().query(
            query_embeddings=q_emb,
            n_results=k,
            include=["documents", "distances"],
        )

    def drop(self):
        self.client.delete_collection(self.collection_name)


# ──────────────────────────────  RAG orchestrator  ────────────────────────────


class PubMedRAG:
    """Wrapper that indexes PubMed text then answers questions."""

    def __init__(
        self,
        chroma_mgr: ChromaDBManager,
        embedding_model_name: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        openai_model: str = "gpt-4.1-nano",
        temperature: float = 0.2,
    ):
        self.chroma_mgr = chroma_mgr
        self.model_name = embedding_model_name
        self.embedder = SentenceTransformer(embedding_model_name)
        self.openai = OpenAI()
        self.llm_model = openai_model
        self.temperature = temperature

    # --------------------------------------------------------------------- #
    # indexing                                                               #
    # --------------------------------------------------------------------- #
    def index_topic(self, email: str, topic: str, retmax: int = 10):
        abstracts = fetch_pubmed_abstracts(email, topic, retmax=retmax)
        chunks = chunk_text(abstracts)
        embeddings = embed_chunks(self.embedder, chunks)
        self.chroma_mgr.add_chunks(chunks, embeddings)

    # --------------------------------------------------------------------- #
    # inference                                                              #
    # --------------------------------------------------------------------- #
    def answer(self, question: str, k: int = 10) -> Dict[str, Any]:
        retrieval = self.chroma_mgr.query(question, self.embedder, k=k)
        context = "\n\n".join(retrieval["documents"][0])

        prompt = (
            "You are a helpful medical research assistant.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )

        resp = self.openai.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        answer_text = resp.choices[0].message.content.strip()

        return {
            "answer": answer_text,
            "retrieval": retrieval,
            "chunks_text": context,
        }


# ──────────────────────────────  chat session  ────────────────────────────────


class ChatSession:
    """High‑level chat interface (keeps history in memory only)."""

    def __init__(self, email: str, topic: str, retmax: int = 10):
        self.id = str(uuid.uuid4())
        self.email = email
        self.topic = topic

        # make a temp collection unique to this chat
        self.chroma_mgr = ChromaDBManager(collection_name=f"pubmed_{self.id}")
        self.rag = PubMedRAG(self.chroma_mgr)

        # fetch & index abstracts
        self.rag.index_topic(email, topic, retmax=retmax)
        logger.info("✅  Abstracts indexed.")

        self.turns: List[Dict[str, Any]] = [
            {"role": "system", "content": "You are a helpful medical research assistant."}
        ]

    def ask(self, question: str, k: int = 10) -> str:
        self.turns.append({"role": "user", "content": question})

        result = self.rag.answer(question, k=k)
        answer_text = result["answer"]

        self.turns.append(
            {
                "role": "assistant",
                "content": answer_text,
                "chunks_text": result["chunks_text"],
            }
        )
        return answer_text

    def close(self):
        """Drop vector store to keep disk tidy."""
        self.chroma_mgr.drop()


# ──────────────────────────────  command‑line driver  ─────────────────────────


def main():
    email = input("NCBI Entrez email: ").strip() or os.getenv("ENTREZ_EMAIL", "")
    if not email:
        print("❌  Need an email for Entrez API. Set ENTREZ_EMAIL or type it now.")
        return

    topic = input("PubMed search topic: ").strip()
    if not topic:
        print("❌  Topic cannot be empty.")
        return

    session = ChatSession(email, topic, retmax=10)

    try:
        while True:
            question = input("\nYou: ").strip()
            if question.lower() in {"quit", "exit"}:
                break
            answer = session.ask(question, k=10)
            print(f"\nAssistant: {answer}")
    finally:
        session.close()
        print("🔚  Session ended (vector store removed).")


if __name__ == "__main__":
    main()
