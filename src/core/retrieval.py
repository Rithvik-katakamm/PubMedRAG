"""
PubMed Data Retrieval Module

Handles fetching abstracts from PubMed using NCBI Entrez.
This is the first step in the RAG pipeline.
"""

from typing import List, Tuple
from Bio import Entrez


def fetch_pubmed_abstracts(
    email: str, 
    topic: str, 
    db: str = "pubmed", 
    retmax: int = 100
) -> Tuple[List[str], str]:
    """Search PubMed for a given term and retrieve abstracts.
    
    Parameters:
    - email: Your email address for NCBI Entrez
    - topic: The term to search for
    - db: The Entrez database to query (default: "pubmed")
    - retmax: Maximum number of articles to retrieve (default: 100)
    
    Returns:
    - id_list: List of PubMed IDs found
    - abstracts_text: Combined abstracts as a single string
    """
    Entrez.email = email
    
    # Step 1: Search for articles
    handle = Entrez.esearch(db=db, term=topic, retmax=retmax)
    record = Entrez.read(handle)
    id_list = record["IdList"]
    
    # Step 2: Fetch abstracts
    if id_list:
        handle = Entrez.efetch(db=db, id=id_list, rettype="abstract", retmode="text")
        abstracts_text = handle.read()
    else:
        abstracts_text = ""
    
    return id_list, abstracts_text