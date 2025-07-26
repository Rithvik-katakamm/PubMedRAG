from src.core import EmbeddingGenerator
import time

def speed_test_embedding_generator():
    """Speed test for the EmbeddingGenerator class."""
    texts = [
        "This is a test sentence.",
        "Another example of a text chunk.",
        "Embedding generation is crucial for NLP tasks.",
        "BioBERT is optimized for biomedical text.",
        "M2 optimizations improve performance significantly."
    ] * 1000  # Increase the number of texts for a more substantial test

    generator = EmbeddingGenerator()
    
    start_time = time.time()
    embeddings = generator.embed_chunks(texts, batch_size=32, show_progress_bar=True)
    end_time = time.time()

    print(f"Generated {len(embeddings)} embeddings in {end_time - start_time:.2f} seconds.")
    print(f"First embedding vector: {embeddings[0]}")