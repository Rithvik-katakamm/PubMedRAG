"""
Vector Database Module

Manages ChromaDB operations with topic-based collections.
Includes automatic FIFO cleanup and M2 performance optimizations.
"""

import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from typing import List, Dict, Optional, Any
import time
from datetime import datetime
import logging

# Suppress ChromaDB warnings about duplicate IDs
logging.getLogger('chromadb').setLevel(logging.ERROR)


class ChromaDBManager:
    """Manages ChromaDB operations with performance optimizations."""
    
    # M2 optimized settings
    MAX_VECTORS_PER_TOPIC = 5000  # Sweet spot for sub-100ms retrieval on M2
    
    def __init__(self, db_path: str = "./data/chroma_db"):
        """Initialize ChromaDB manager with M2 optimized settings."""
        self.db_path = db_path
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            ),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE
        )
        self._embedding_generator = None
    
    def set_embedding_generator(self, generator):
        """Set the embedding generator for retrieval."""
        self._embedding_generator = generator
    
    def get_or_create_collection(self, topic: str):
        """Get or create a collection for a specific topic."""
        # Clean topic name for collection naming - NO timestamp for persistence
        collection_name = topic.lower().replace(" ", "_")[:50]
        
        try:
            collection = self.client.get_collection(name=collection_name)
        except:
            # Create with HNSW index for fast similarity search
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine", "topic": topic}
            )
        
        return collection
    
    def store_chunks(
        self, 
        topic: str,
        chunk_texts: List[str], 
        embeddings: List[List[float]], 
        pubmed_ids: List[str]
    ) -> Dict[str, Any]:
        """Store text chunks and embeddings, with automatic FIFO cleanup.
        
        Returns metrics about the storage operation.
        """
        start_time = time.time()
        collection = self.get_or_create_collection(topic)
        
        # Check current collection size
        current_count = collection.count()
        new_count = len(chunk_texts)
        
        # FIFO cleanup if needed
        if current_count + new_count > self.MAX_VECTORS_PER_TOPIC:
            # Calculate how many to remove
            to_remove = (current_count + new_count) - self.MAX_VECTORS_PER_TOPIC
            
            # Get oldest IDs and remove them
            all_data = collection.get(limit=to_remove)
            if all_data["ids"]:
                collection.delete(ids=all_data["ids"])
        
        # Prepare metadata for each chunk
        metadatas = []
        for i, (chunk, pmid) in enumerate(zip(chunk_texts, pubmed_ids)):
            metadatas.append({
                "chunk_index": i,
                "pubmed_id": pmid,
                "timestamp": datetime.now().isoformat(),
                "char_count": len(chunk)
            })
        
        # Generate IDs based on current max ID
        current_data = collection.get(limit=1)
        if current_data["ids"]:
            max_id = max([int(id_) for id_ in current_data["ids"] if id_.isdigit()], default=0)
        else:
            max_id = 0
        
        ids = [str(max_id + i + 1) for i in range(len(chunk_texts))]
        
        # Add to collection
        collection.add(
            ids=ids,
            documents=chunk_texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        storage_time = time.time() - start_time
        
        return {
            "storage_time_ms": int(storage_time * 1000),
            "chunks_stored": new_count,
            "total_chunks": collection.count(),
            "cleanup_performed": current_count + new_count > self.MAX_VECTORS_PER_TOPIC
        }
    
    def retrieve_chunks(self, topic: str, query: str, k: int = 10) -> Dict[str, Any]:
        """Retrieve relevant chunks with performance metrics."""
        start_time = time.time()
        
        collection = self.get_or_create_collection(topic)
        
        # Generate query embedding
        embedding_start = time.time()
        query_embedding = self._embedding_generator.embed_chunks([query])[0]
        embedding_time = time.time() - embedding_start
        
        # Perform retrieval
        retrieval_start = time.time()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, collection.count() or 1),
            include=["documents", "metadatas", "distances"]
        )
        retrieval_time = time.time() - retrieval_start
        
        total_time = time.time() - start_time
        
        return {
            "documents": results["documents"],
            "metadatas": results["metadatas"],
            "distances": results["distances"],
            "ids": results["ids"],
            "metrics": {
                "embedding_time_ms": int(embedding_time * 1000),
                "retrieval_time_ms": int(retrieval_time * 1000),
                "total_time_ms": int(total_time * 1000),
                "chunks_retrieved": len(results["documents"][0]) if results["documents"] else 0
            }
        }
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about all collections."""
        stats = {}
        for coll in self.client.list_collections():
            stats[coll.name] = coll.count()
        return stats
    
    def clear_topic(self, topic: str):
        """Clear all data for a specific topic."""
        collection_name = topic.lower().replace(" ", "_")[:50]
        try:
            self.client.delete_collection(name=collection_name)
            return True
        except:
            return False
    
    def clear_all(self):
        """Clear all collections."""
        for coll in self.client.list_collections():
            self.client.delete_collection(name=coll.name)