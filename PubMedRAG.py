

"""
PubMed RAG System

This module provides functionality to:
1. Fetch abstracts from PubMed
2. Process and chunk text data
3. Generate embeddings for text chunks
4. Store and retrieve data using ChromaDB
5. Answer medical questions using RAG (Retrieval-Augmented Generation)
6. Persist data in a PostgreSQL database
"""

import os
import logging
from typing import List, Tuple, Dict, Union, Optional, Any

# Third-party imports
from Bio import Entrez
import psycopg2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from openai import OpenAI
from db.db_utils import get_connection, store_interaction
from dotenv import load_dotenv
load_dotenv('env_variables.env')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# PubMed data fetching
def fetch_pubmed_abstracts(
    email: str, 
    topic: str, 
    db: str = "pubmed", 
    retmax: int = 10
) -> Tuple[List[str], str]:
    """Search PubMed for a given term and retrieve abstracts.
    
    Parameters:
    - email: Your email address for NCBI Entrez
    - search_query: The term to search for
    - db: The Entrez database to query (default: "pubmed")
    - retmax: Maximum number of articles to retrieve (default: 10)
    
    Returns:
    - id_list: List of PubMed IDs found
    - abstracts_text: Combined abstracts as a single string
    """
    Entrez.email = email
    
    # Step 1: Search for articles
    handle = Entrez.esearch(db=db, term=topic, retmax=retmax)
    record = Entrez.read(handle)
    id_list = record["IdList"]
    logger.info(f"Found {len(id_list)} articles for '{topic}'")
    
    # Step 2: Fetch abstracts
    handle = Entrez.efetch(db=db, id=id_list, rettype="abstract", retmode="text")
    abstracts_text = handle.read()
    
    return id_list, abstracts_text

# Text processing
def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 50,
    separators: Optional[List[str]] = None
) -> List[str]:
    """Split a long text into overlapping chunks using LangChain's RecursiveCharacterTextSplitter.
    
    Parameters:
    - text: The input string to split
    - chunk_size: Maximum number of characters per chunk
    - chunk_overlap: Number of characters to overlap between chunks
    - separators: List of separators to use in order of priority
    
    Returns:
    - List of text chunks
    """
    if separators is None:
        separators = ["\n\n", "\n", ".", "!", "?", " ", ""]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    
    docs = splitter.create_documents([text])
    chunk_texts = [doc.page_content for doc in docs]
    
    logger.info(f"Chunks created: {len(chunk_texts)}")
    if chunk_texts:
        logger.debug(f"Sample chunk: {chunk_texts[0][:100]}...")
    
    return chunk_texts

# Embedding generation
def embed_chunks(
    model_or_name: Union[str, SentenceTransformer],
    texts: List[str],
    show_progress_bar: bool = True
) -> List[List[float]]:
    """Generate embeddings for a list of text chunks.
    
    Parameters:
    - model_or_name: Either a SentenceTransformer instance or a model name
    - texts: List of strings to embed
    - show_progress_bar: Whether to display a progress bar
    
    Returns:
    - embeddings: List of embedding vectors
    """
    # Load model if needed
    model = (
        model_or_name
        if not isinstance(model_or_name, str)
        else SentenceTransformer(model_or_name)
    )
    
    # Generate embeddings
    embeddings = model.encode(texts, show_progress_bar=show_progress_bar)
    
    # Log info
    n = len(embeddings)
    dim = embeddings.shape[1] if hasattr(embeddings, "shape") else len(embeddings[0])
    logger.info(f"Generated {n} embeddings, each of size {dim}")
    
    return embeddings

# Vector database functionality
class ChromaDBManager:
    """Class to manage ChromaDB operations."""
    
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "pubmed_collection"):
        """Initialize ChromaDB manager.
        
        Parameters:
        - db_path: Path to store the ChromaDB data
        - collection_name: Name of the collection to use
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE
        )
        
    def get_collection(self):
        """Get the collection object."""
        return self.client.get_collection(name=self.collection_name)
    
    def get_or_create_collection(self):
        """Get or create a collection."""
        return self.client.get_or_create_collection(name=self.collection_name)
    
    def store_chunks(self, chunk_texts: List[str], embeddings: List[List[float]], 
                    source: str = "pubmed", metadata: Optional[List[Dict]] = None):
        """Store text chunks and their embeddings in ChromaDB.
        
        Parameters:
        - chunk_texts: List of text chunks
        - embeddings: List of embedding vectors
        - source: Source of the data
        - metadata: Additional metadata for each chunk
        """
        collection = self.get_or_create_collection()
        
        # Prepare IDs and metadata
        ids = [str(i) for i in range(len(chunk_texts))]
        
        if metadata is None:
            metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunk_texts))]
        else:
            metadatas = metadata
        
        # Add documents, embeddings, and metadata to the collection
        collection.add(
            ids=ids,
            documents=chunk_texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        logger.info(f"Stored {collection.count()} chunks in ChromaDB")
    
    def retrieve_chunks(self, query: str, model, k: int = 10):
        """Retrieve chunks from ChromaDB based on query similarity.
        
        Parameters:
        - query: Query string
        - model: Embedding model to use
        - k: Number of results to retrieve
        
        Returns:
        - Query result from ChromaDB
        """
        collection = self.get_collection()
        q_emb = model.encode([query]).tolist()
        
        result = collection.query(
            query_embeddings=q_emb,
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Log results
        for idx, (doc, meta, dist) in enumerate(
            zip(result["documents"][0], result["metadatas"][0], result["distances"][0]), 1
        ):
            logger.debug(f"Result {idx}: [score={dist:.4f}] {meta}")
            logger.debug(f"{doc[:100]}...\n")
        
        return result

# Question answering functionality
class PubMedRAG:
    """Class for medical question answering using PubMed data."""
    
    def __init__(
        self, 
        db_manager: ChromaDBManager,
        embedding_model: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        openai_api_key: Optional[str] = None
    ):
        """Initialize the RAG system.
        
        Parameters:
        - db_manager: ChromaDB manager instance
        - embedding_model: Model to use for embeddings
        - openai_api_key: OpenAI API key (optional, will use env var if not provided)
        """
        self.db_manager = db_manager
        self.model = SentenceTransformer(embedding_model)
        
        # Initialize OpenAI client
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        else:
            self.openai_client = OpenAI()
    
    def process_pubmed_data(self, email: str, search_query: str, retmax: int = 10):
        """Fetch, process, and store PubMed data.
        
        Parameters:
        - email: Email for NCBI Entrez
        - search_query: Query for PubMed search
        - retmax: Maximum number of articles to retrieve
        """
        # Fetch data
        id_list, abstracts_text = fetch_pubmed_abstracts(email, search_query, retmax=retmax)
        
        # Process text
        chunk_texts = chunk_text(abstracts_text)
        
        # Generate embeddings
        embeddings = embed_chunks(self.model, chunk_texts)
        
        # Store in ChromaDB
        self.db_manager.store_chunks(chunk_texts, embeddings.tolist())
        
        return id_list, abstracts_text, chunk_texts
    
    def retrieve_chunks(self, query: str, k: int = 10):
        """Retrieve relevant chunks for a query.
        
        Parameters:
        - query: Query string
        - k: Number of results to retrieve
        
        Returns:
        - Retrieved chunks and metadata
        """
        return self.db_manager.retrieve_chunks(query, self.model, k)
    
    def answer_question(self, query: str, k: int = 10, temperature: float = 0.2, model: str = "gpt-4.1-nano"):
        """Answer a medical question using retrieved context.
        
        Parameters:
        - query: Question to answer
        - k: Number of chunks to retrieve
        - temperature: Temperature for OpenAI API
        - model: Model to use for answering
        
        Returns:
        - Answer string
        - Retrieved chunks
        """
        # Retrieve relevant chunks
        retrieval = self.retrieve_chunks(query, k)
        
        # Combine chunks into context
        context = "\n\n".join(retrieval["documents"][0])
        
        # Create prompt
        prompt = f"""You are a medical research assistant. Use the following PubMed abstracts to answer the question.

{context}

Question: {query}
Answer:"""
        
        # Generate answer
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful medical research assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        
        answer = response.choices[0].message.content
        logger.info(f"Generated answer for query: '{query}'")
        
        return answer, retrieval

def main():
    """Main function to demonstrate usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="PubMed RAG System")
    parser.add_argument("--email", required=True, help="Email for NCBI Entrez")
    parser.add_argument("--topic", required=True, help="Query to search PubMed")
    parser.add_argument("--query", default="general", help="LLM query")
    parser.add_argument("--retmax", type=int, default=10, help="Maximum articles to retrieve")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve")
    
    args = parser.parse_args()
    
 
    
    # Initialize RAG system
    db_manager = ChromaDBManager()
    rag_system = PubMedRAG(db_manager)
    
    # Process data
    rag_system.process_pubmed_data(args.email, args.topic, retmax=args.retmax)
    
    # Answer question
    answer, retrieval = rag_system.answer_question(args.query, k=args.k)
    
    # Print answer
    print("Answer:")
    print(answer)
    
    # Store retrieved data
    retrieved_chunks_str = "\n\n".join(retrieval["documents"][0])
    store_interaction(
    args.email,
    args.topic,
    args.query,
    retrieved_chunks_str,
    answer
)
    
if __name__ == "__main__":
    main()