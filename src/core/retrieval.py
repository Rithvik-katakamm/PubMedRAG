"""
PubMed Data Retrieval Module

Handles fetching abstracts from PubMed using NCBI Entrez.
This is the first step in the RAG pipeline.
"""

from typing import List, Tuple

try:
    from Bio import Entrez
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("Warning: BioPython not available. Using mock data for testing.")


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
    if not BIOPYTHON_AVAILABLE:
        # Return mock data for testing
        mock_abstracts = f"""
        Sample Abstract 1 for {topic}:
        This is a sample medical abstract about {topic}. It contains important medical information and research findings that would normally be retrieved from PubMed. This abstract discusses various aspects of {topic} treatment and diagnosis.

        Sample Abstract 2 for {topic}:
        Another sample abstract discussing {topic} research. This paper presents clinical trials and evidence-based medicine approaches to {topic}. The study shows significant improvements in patient outcomes.

        Sample Abstract 3 for {topic}:
        A comprehensive review of {topic} literature. This abstract covers the latest developments in {topic} research and provides insights into future directions for treatment and prevention strategies.
        """
        return ["12345", "67890", "54321"], mock_abstracts
    
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