"""
Main RAG Engine Module

Orchestrates the complete RAG pipeline:
Retrieval → Chunking → Embedding → Storage → Question Answering
"""

import time
from typing import List, Tuple, Dict, Optional, Any
from openai import OpenAI

from .retrieval import fetch_pubmed_abstracts
from .chunking import chunk_text
from .embeddings import EmbeddingGenerator
from .vector_db import ChromaDBManager


class PubMedRAG:
    """Main RAG system for medical question answering using PubMed data."""
    
    def __init__(
        self, 
        db_manager: Optional[ChromaDBManager] = None,
        openai_api_key: Optional[str] = None
    ):
        """Initialize the RAG system.
        
        Parameters:
        - db_manager: ChromaDB manager instance (creates one if not provided)
        - openai_api_key: OpenAI API key (uses env var if not provided)
        """
        self.db_manager = db_manager or ChromaDBManager()
        self.embedding_generator = EmbeddingGenerator()
        self.db_manager.set_embedding_generator(self.embedding_generator)
        
        # Initialize OpenAI client
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        else:
            self.openai_client = OpenAI()
    
    def process_pubmed_data(
        self, 
        email: str, 
        topic: str, 
        retmax: int = 100
    ) -> Dict[str, Any]:
        """Fetch, process, and store PubMed data with performance metrics.
        
        Returns dictionary with results and timing metrics.
        """
        metrics = {}
        
        # Fetch data from PubMed
        fetch_start = time.time()
        id_list, abstracts_text = fetch_pubmed_abstracts(email, topic, retmax=retmax)
        metrics["fetch_time_ms"] = int((time.time() - fetch_start) * 1000)
        
        if not abstracts_text:
            return {
                "id_list": [],
                "chunk_count": 0,
                "metrics": metrics,
                "error": "No abstracts found for this topic"
            }
        
        # Chunk the text
        chunk_start = time.time()
        chunk_texts = chunk_text(abstracts_text)
        metrics["chunking_time_ms"] = int((time.time() - chunk_start) * 1000)
        
        # Generate embeddings
        embed_start = time.time()
        embeddings = self.embedding_generator.embed_chunks(chunk_texts)
        metrics["embedding_time_ms"] = int((time.time() - embed_start) * 1000)
        
        # Prepare PubMed IDs for each chunk (distribute IDs across chunks)
        pubmed_ids = []
        if id_list:
            # Simple distribution: assign PubMed IDs proportionally to chunks
            ids_per_chunk = max(1, len(id_list) // len(chunk_texts))
            for i, chunk in enumerate(chunk_texts):
                # Get the corresponding PubMed ID
                id_index = min(i // ids_per_chunk, len(id_list) - 1)
                pubmed_ids.append(id_list[id_index])
        else:
            pubmed_ids = ["unknown"] * len(chunk_texts)
        
        # Store in ChromaDB
        storage_metrics = self.db_manager.store_chunks(
            topic=topic,
            chunk_texts=chunk_texts,
            embeddings=embeddings,
            pubmed_ids=pubmed_ids
        )
        
        metrics["storage_time_ms"] = storage_metrics["storage_time_ms"]
        metrics["total_time_ms"] = int((time.time() - fetch_start) * 1000)
        
        return {
            "id_list": id_list,
            "chunk_count": len(chunk_texts),
            "abstracts_count": len(id_list),
            "total_chunks_in_db": storage_metrics["total_chunks"],
            "metrics": metrics
        }
    
    def answer_question(
        self, 
        topic: str,
        query: str, 
        k: int = 10, 
        temperature: float = 0.2, 
        model: str = "gpt-4o-mini"
    ) -> Tuple[str, Dict[str, Any]]:
        """Answer a medical question using retrieved context.
        
        Returns:
        - answer: Generated answer string
        - retrieval_info: Dictionary with retrieval details and metrics
        """
        # Retrieve relevant chunks
        retrieval_result = self.db_manager.retrieve_chunks(topic, query, k)
        
        if not retrieval_result["documents"] or not retrieval_result["documents"][0]:
            return "I don't have enough information to answer this question.", retrieval_result
        
        # Combine chunks into context
        context = "\n\n".join(retrieval_result["documents"][0])
        
        # Create prompt
        prompt = f"""You are a helpful medical research assistant. Answer the following question based ONLY on the provided research context from PubMed abstracts.

Context from PubMed research:
{context}

Question: {query}

Instructions:
- Base your answer solely on the provided context
- Be concise but comprehensive
- Cite specific findings when possible
- If the context doesn't contain enough information, say so

Answer:"""
        
        # Generate answer
        generation_start = time.time()
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a medical research assistant that provides accurate, evidence-based answers."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=1000
        )
        generation_time = int((time.time() - generation_start) * 1000)
        
        answer = response.choices[0].message.content
        
        # Add generation metrics to retrieval info
        retrieval_result["metrics"]["generation_time_ms"] = generation_time
        retrieval_result["metrics"]["total_rag_time_ms"] = (
            retrieval_result["metrics"]["total_time_ms"] + generation_time
        )
        
        # Add token usage info
        retrieval_result["token_usage"] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "model": model
        }
        
        return answer, retrieval_result