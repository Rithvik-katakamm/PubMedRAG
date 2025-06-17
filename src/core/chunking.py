"""
Text Chunking Module

Splits long text into overlapping chunks for better retrieval.
Uses LangChain's RecursiveCharacterTextSplitter with your chosen parameters.
"""

from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 50,
    separators: Optional[List[str]] = None
) -> List[str]:
    """Split a long text into overlapping chunks.
    
    Parameters:
    - text: The input string to split
    - chunk_size: Maximum number of characters per chunk (default: 1000)
    - chunk_overlap: Number of characters to overlap between chunks (default: 50)
    - separators: List of separators to use in order of priority
    
    Returns:
    - List of text chunks
    """
    if not text.strip():
        return []
        
    if separators is None:
        separators = ["\n\n", "\n", ".", "!", "?", " ", ""]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    
    docs = splitter.create_documents([text])
    chunk_texts = [doc.page_content for doc in docs]
    
    return chunk_texts