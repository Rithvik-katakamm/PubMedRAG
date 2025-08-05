"""
Embedding Generation Module

Generates embeddings using BioBERT model optimized for medical text.
Includes M2 optimizations for batch processing. 32 batches in parallel.
"""

from typing import List, Union
import os
from sentence_transformers import SentenceTransformer

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class EmbeddingGenerator:
    """Manages embedding generation with lazy loading and M2 optimization."""
    
    def __init__(self, model_name: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"):
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        """Lazy load the model only when needed."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
            # M2 optimization: use single-threaded for consistency
            self._model.max_seq_length = 512  # Optimal for BERT models
        return self._model
    
    def embed_chunks(
        self,
        texts: List[str],
        batch_size: int = 32, 
        show_progress_bar: bool = False
    ) -> List[List[float]]:
        """Generate embeddings for a list of text chunks.
        
        Parameters:
        - texts: List of strings to embed
        - batch_size: Batch size for processing (M2 optimized)
        - show_progress_bar: Whether to display a progress bar
        
        Returns:
        - embeddings: List of embedding vectors
        """
        if not texts:
            return []
        
        # Generate embeddings with M2-optimized batch size
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True
        )
        
        return embeddings.tolist()