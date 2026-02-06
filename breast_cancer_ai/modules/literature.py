"""
Literature Mining Module
Fetches relevant PubMed papers using Biopython Entrez
"""

from Bio import Entrez
import time

# NCBI API Key for higher rate limits (10 requests/sec vs 3/sec)
NCBI_API_KEY = "ae83b1da74148ccbacc801302448d41ae708"

# ============================================================
# PREDEFINED PUBMED QUERIES (Comprehensive Breast Cancer Research)
# ============================================================

PREDEFINED_QUERIES = {
    # === HISTOPATHOLOGY & IMAGING ===
    "Breast Histopathology AI": '("breast cancer"[Title] AND "histopathology"[Title/Abstract] AND ("deep learning" OR "CNN" OR "neural network"))',
    "BreakHis Dataset Studies": '("BreakHis" OR "breast histopathology images") AND classification',
    "GradCAM Explainability": '("Grad-CAM" OR "GradCAM" OR "class activation") AND ("breast" OR "cancer") AND explainability',
    "Digital Pathology Review": '"digital pathology"[Title] AND "breast cancer"[Title/Abstract] AND (diagnosis OR classification)',
    
    # === GENOMICS & BIOMARKERS ===
    "BRCA1 BRCA2 Mutations": '("BRCA1"[Title] OR "BRCA2"[Title]) AND "breast cancer" AND (mutation OR variant) AND prognosis',
    "Triple Negative Biomarkers": '"triple negative breast cancer"[Title] AND biomarkers AND (treatment OR therapy)',
    "HER2 Targeted Therapy": '("HER2" OR "ERBB2")[Title] AND "breast cancer" AND ("targeted therapy" OR trastuzumab)',
    "Gene Expression Profiling": '"gene expression"[Title] AND "breast cancer" AND (subtype OR classification OR prognosis)',
    "TCGA Breast Cancer": '"TCGA"[Title/Abstract] AND "breast cancer" AND ("RNA-seq" OR "gene expression" OR genomic)',
    
    # === MACHINE LEARNING & AI ===
    "ML Breast Cancer Diagnosis": '"machine learning"[Title] AND "breast cancer"[Title] AND (diagnosis OR detection OR prediction)',
    "Deep Learning Mammography": '"deep learning"[Title] AND (mammography OR mammogram) AND "breast cancer"',
    "Random Forest Cancer": '"random forest"[Title/Abstract] AND "breast cancer" AND (classification OR prediction)',
    "XAI Medical Imaging": '("explainable AI" OR "interpretable")[Title] AND ("breast cancer" OR "medical imaging")',
    
    # === MULTIMODAL & INTEGRATION ===
    "Multimodal Cancer Diagnosis": '"multimodal"[Title] AND "breast cancer" AND (imaging OR genomic OR pathology)',
    "Radiogenomics Breast": '("radiogenomics" OR "imaging genomics")[Title/Abstract] AND "breast cancer"',
    "Integrated Omics Analysis": '("multi-omics" OR "integrated analysis")[Title] AND "breast cancer" AND prognosis',
    
    # === CLINICAL & TREATMENT ===
    "Breast Cancer Subtypes": '"breast cancer subtypes"[Title] AND (luminal OR basal OR "molecular subtype")',
    "Immunotherapy Breast Cancer": '"immunotherapy"[Title] AND "breast cancer" AND (PD-1 OR PD-L1 OR checkpoint)',
    "Neoadjuvant Chemotherapy": '"neoadjuvant"[Title] AND "breast cancer" AND (response OR pathological)',
    
    # === RECENT ADVANCES ===
    "Foundation Models Pathology": '("foundation model" OR "large language model" OR "vision transformer")[Title/Abstract] AND pathology',
    "AI Cancer Screening 2024": '("artificial intelligence" OR "AI")[Title] AND "breast cancer" AND screening AND 2024[Date - Publication]',
}

# Quick access list for UI dropdown
QUERY_OPTIONS = list(PREDEFINED_QUERIES.keys())


def get_query(query_name):
    """Get the actual PubMed query string from a predefined query name"""
    return PREDEFINED_QUERIES.get(query_name, query_name)


def fetch_pubmed(query, max_results=5, email="ai@example.com"):
    """
    Fetch PubMed papers based on biomedical text query
    
    Args:
        query (str): Biomedical search query (e.g., "breast cancer BRCA1 TP53")
        max_results (int): Maximum number of papers to retrieve
        email (str): Email for NCBI Entrez (required by NCBI)
    
    Returns:
        list: List of dicts with keys: title, authors, abstract, pmid, journal
    """
    Entrez.email = email
    Entrez.api_key = NCBI_API_KEY
    papers = []
    
    try:
        # Search PubMed for relevant paper IDs
        search_handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results,
            sort="relevance"
        )
        search_results = Entrez.read(search_handle)
        search_handle.close()
        
        id_list = search_results.get("IdList", [])
        
        if not id_list:
            return [{
                "title": "No results found",
                "authors": "",
                "abstract": f"No PubMed papers found for query: {query}",
                "pmid": "N/A",
                "journal": ""
            }]
        
        # Fetch detailed information for each paper
        time.sleep(0.5)  # Be nice to NCBI servers
        fetch_handle = Entrez.efetch(
            db="pubmed",
            id=id_list,
            rettype="medline",
            retmode="xml"
        )
        papers_data = Entrez.read(fetch_handle)
        fetch_handle.close()
        
        # Extract relevant information
        for paper in papers_data['PubmedArticle']:
            try:
                article = paper['MedlineCitation']['Article']
                
                # Extract title
                title = article.get('ArticleTitle', 'No title available')
                
                # Extract authors
                authors_list = article.get('AuthorList', [])
                authors = ", ".join([
                    f"{author.get('LastName', '')} {author.get('Initials', '')}"
                    for author in authors_list[:3]  # First 3 authors
                ])
                if len(authors_list) > 3:
                    authors += " et al."
                
                # Extract abstract
                abstract_text = article.get('Abstract', {})
                if abstract_text:
                    abstract_parts = abstract_text.get('AbstractText', [])
                    if isinstance(abstract_parts, list):
                        abstract = " ".join([str(part) for part in abstract_parts])
                    else:
                        abstract = str(abstract_parts)
                else:
                    abstract = "Abstract not available"
                
                # Extract PMID
                pmid = str(paper['MedlineCitation']['PMID'])
                
                # Extract journal
                journal = article.get('Journal', {}).get('Title', 'Unknown journal')
                
                papers.append({
                    "title": title,
                    "authors": authors,
                    "abstract": abstract[:500] + "..." if len(abstract) > 500 else abstract,
                    "pmid": pmid,
                    "journal": journal
                })
                
            except Exception as e:
                # Skip papers with parsing errors
                continue
        
        if not papers:
            papers = [{
                "title": "Error parsing results",
                "authors": "",
                "abstract": "Papers found but could not be parsed",
                "pmid": "N/A",
                "journal": ""
            }]
            
    except Exception as e:
        # Handle API errors gracefully
        papers = [{
            "title": "API Error",
            "authors": "",
            "abstract": f"Error fetching PubMed data: {str(e)}. Please check your internet connection or try a different query.",
            "pmid": "N/A",
            "journal": ""
        }]
    
    return papers


if __name__ == "__main__":
    # Test the module
    results = fetch_pubmed("breast cancer BRCA1 TP53", max_results=3)
    for i, paper in enumerate(results, 1):
        print(f"\n{i}. {paper['title']}")
        print(f"   PMID: {paper['pmid']}")
        print(f"   {paper['abstract'][:200]}...")
