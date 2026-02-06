"""
Literature Mining with Named Entity Recognition (NER)
Extends the base literature module with BioBERT-based entity extraction
"""

import re
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict

# Import base literature functions
from .literature import fetch_pubmed, PREDEFINED_QUERIES, QUERY_OPTIONS, get_query

# Try to import NER libraries
try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

# Biomedical NER model options
NER_MODELS = {
    'biobert': 'dmis-lab/biobert-base-cased-v1.1',
    'pubmedbert': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
    'scibert': 'allenai/scibert_scivocab_uncased',
}

DEFAULT_NER_MODEL = 'biobert'

# Entity types to extract
ENTITY_TYPES = {
    'GENE': ['gene', 'genes', 'protein', 'proteins', 'receptor', 'enzyme'],
    'DRUG': ['drug', 'drugs', 'treatment', 'therapy', 'inhibitor', 'agent'],
    'DISEASE': ['cancer', 'tumor', 'carcinoma', 'neoplasm', 'malignant', 'disease'],
    'MUTATION': ['mutation', 'variant', 'deletion', 'amplification', 'polymorphism'],
    'PATHWAY': ['pathway', 'signaling', 'cascade', 'mechanism'],
    'BIOMARKER': ['biomarker', 'marker', 'indicator', 'expression'],
}

# Known biomedical entities (curated list for pattern matching)
KNOWN_GENES = [
    'BRCA1', 'BRCA2', 'TP53', 'HER2', 'ERBB2', 'ESR1', 'PGR', 'PIK3CA',
    'AKT1', 'PTEN', 'EGFR', 'MYC', 'CCND1', 'CDH1', 'RB1', 'CDKN2A',
    'MDM2', 'KRAS', 'BRAF', 'ATM', 'CHEK2', 'PALB2', 'RAD51', 'FGFR1',
    'MKI67', 'Ki67', 'Ki-67', 'BCL2', 'BAX', 'VEGF', 'mTOR', 'CDK4', 'CDK6'
]

KNOWN_DRUGS = [
    'Tamoxifen', 'Trastuzumab', 'Herceptin', 'Letrozole', 'Anastrozole',
    'Paclitaxel', 'Doxorubicin', 'Cyclophosphamide', 'Carboplatin',
    'Pertuzumab', 'Palbociclib', 'Olaparib', 'Everolimus', 'Fulvestrant',
    'Exemestane', 'Bevacizumab', 'Neratinib', 'Tucatinib', 'Capecitabine'
]

KNOWN_DISEASES = [
    'breast cancer', 'invasive ductal carcinoma', 'invasive lobular carcinoma',
    'DCIS', 'ductal carcinoma in situ', 'triple negative breast cancer',
    'TNBC', 'HER2-positive', 'ER-positive', 'metastatic breast cancer',
    'inflammatory breast cancer', 'Paget disease', 'phyllodes tumor'
]


# ============================================================
# RULE-BASED NER (No deep learning dependencies)
# ============================================================

class RuleBasedNER:
    """
    Rule-based Named Entity Recognition using regex patterns and curated lists
    Works without any deep learning dependencies
    """
    
    def __init__(self):
        """Initialize with curated entity lists"""
        self.genes = set(g.upper() for g in KNOWN_GENES)
        self.drugs = set(d.lower() for d in KNOWN_DRUGS)
        self.diseases = [d.lower() for d in KNOWN_DISEASES]
        
        # Regex patterns
        self.gene_pattern = re.compile(
            r'\b([A-Z][A-Z0-9]{1,5}[0-9]?)\b'  # Gene symbols like BRCA1, TP53
        )
        self.mutation_pattern = re.compile(
            r'\b([A-Z]\d+[A-Z]|c\.\d+[A-Z]>[A-Z]|p\.[A-Z][a-z]{2}\d+[A-Z][a-z]{2})\b'
        )
        self.pathway_pattern = re.compile(
            r'\b(\w+(?:\s*[-/]\s*\w+)*\s+(?:pathway|signaling|cascade))\b',
            re.IGNORECASE
        )
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract biomedical entities from text
        
        Args:
            text: Input text (abstract, title, etc.)
        
        Returns:
            List of entity dicts with 'text', 'type', 'start', 'end'
        """
        entities = []
        text_lower = text.lower()
        text_upper = text.upper()
        
        # Extract genes
        for match in self.gene_pattern.finditer(text):
            if match.group(1).upper() in self.genes:
                entities.append({
                    'text': match.group(1),
                    'type': 'GENE',
                    'start': match.start(),
                    'end': match.end()
                })
        
        # Also check for genes in lowercase in text
        words = text.split()
        for i, word in enumerate(words):
            if word.upper() in self.genes:
                entities.append({
                    'text': word.upper(),
                    'type': 'GENE',
                    'start': text.find(word),
                    'end': text.find(word) + len(word)
                })
        
        # Extract drugs (case-insensitive)
        for drug in self.drugs:
            idx = text_lower.find(drug)
            while idx != -1:
                entities.append({
                    'text': text[idx:idx+len(drug)],
                    'type': 'DRUG',
                    'start': idx,
                    'end': idx + len(drug)
                })
                idx = text_lower.find(drug, idx + 1)
        
        # Extract diseases
        for disease in self.diseases:
            idx = text_lower.find(disease)
            while idx != -1:
                entities.append({
                    'text': text[idx:idx+len(disease)],
                    'type': 'DISEASE',
                    'start': idx,
                    'end': idx + len(disease)
                })
                idx = text_lower.find(disease, idx + 1)
        
        # Extract mutations
        for match in self.mutation_pattern.finditer(text):
            entities.append({
                'text': match.group(1),
                'type': 'MUTATION',
                'start': match.start(),
                'end': match.end()
            })
        
        # Extract pathways
        for match in self.pathway_pattern.finditer(text):
            entities.append({
                'text': match.group(1),
                'type': 'PATHWAY',
                'start': match.start(),
                'end': match.end()
            })
        
        # Deduplicate by position
        seen = set()
        unique_entities = []
        for ent in entities:
            key = (ent['start'], ent['end'], ent['type'])
            if key not in seen:
                seen.add(key)
                unique_entities.append(ent)
        
        return sorted(unique_entities, key=lambda x: x['start'])
    
    def get_entity_counts(self, entities: List[Dict]) -> Dict[str, int]:
        """Count entities by type"""
        counts = Counter(e['type'] for e in entities)
        return dict(counts)
    
    def get_unique_entities(self, entities: List[Dict]) -> Dict[str, List[str]]:
        """Get unique entity texts by type"""
        by_type = defaultdict(set)
        for e in entities:
            by_type[e['type']].add(e['text'])
        return {k: sorted(list(v)) for k, v in by_type.items()}


# ============================================================
# TRANSFORMER-BASED NER (BioBERT)
# ============================================================

class BioBERTNER:
    """
    BioBERT-based Named Entity Recognition
    Uses Hugging Face transformers
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize BioBERT NER
        
        Args:
            model_name: Model identifier or path
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required. Install with: pip install transformers")
        
        # Use biomedical NER model
        if model_name is None:
            model_name = "dmis-lab/biobert-v1.1"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple"
            )
            self.available = True
        except Exception as e:
            print(f"Warning: Could not load NER model: {e}")
            self.available = False
            self.fallback = RuleBasedNER()
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract named entities using BioBERT
        
        Args:
            text: Input text
        
        Returns:
            List of entity dicts
        """
        if not self.available:
            return self.fallback.extract_entities(text)
        
        try:
            # Run NER pipeline
            results = self.ner_pipeline(text)
            
            entities = []
            for r in results:
                # Map model labels to our entity types
                entity_type = self._map_label(r.get('entity_group', ''))
                
                entities.append({
                    'text': r.get('word', ''),
                    'type': entity_type,
                    'start': r.get('start', 0),
                    'end': r.get('end', 0),
                    'score': r.get('score', 0.0)
                })
            
            return entities
            
        except Exception as e:
            print(f"NER error, using fallback: {e}")
            return self.fallback.extract_entities(text)
    
    def _map_label(self, label: str) -> str:
        """Map model labels to our entity types"""
        label = label.upper()
        
        if 'GENE' in label or 'PROTEIN' in label:
            return 'GENE'
        elif 'DRUG' in label or 'CHEM' in label:
            return 'DRUG'
        elif 'DISEASE' in label:
            return 'DISEASE'
        elif 'CELL' in label:
            return 'CELL_TYPE'
        else:
            return 'BIOMARKER'


# ============================================================
# LITERATURE MINING WITH NER
# ============================================================

class LiteratureMiner:
    """
    Complete literature mining pipeline with NER
    Combines PubMed fetching with entity extraction
    """
    
    def __init__(self, use_biobert: bool = False):
        """
        Initialize Literature Miner
        
        Args:
            use_biobert: Use BioBERT NER (requires transformers)
        """
        if use_biobert and TRANSFORMERS_AVAILABLE:
            try:
                self.ner = BioBERTNER()
            except:
                self.ner = RuleBasedNER()
        else:
            self.ner = RuleBasedNER()
        
        self.papers_cache = {}
        self.entities_cache = {}
    
    def search_and_extract(
        self,
        query: str,
        max_results: int = 10,
        email: str = "ai@example.com"
    ) -> Dict:
        """
        Search PubMed and extract entities from results
        
        Args:
            query: Search query (can be predefined name or custom)
            max_results: Maximum papers to fetch
            email: Email for NCBI
        
        Returns:
            Dict with papers, entities, and statistics
        """
        # Check if query is a predefined name
        actual_query = get_query(query)
        
        # Fetch papers
        papers = fetch_pubmed(actual_query, max_results=max_results, email=email)
        
        # Extract entities from each paper
        all_entities = []
        papers_with_entities = []
        
        for paper in papers:
            if paper['pmid'] == 'N/A':
                continue
            
            # Combine title and abstract for extraction
            text = f"{paper['title']} {paper['abstract']}"
            
            # Extract entities
            entities = self.ner.extract_entities(text)
            
            # Add paper reference to entities
            for e in entities:
                e['pmid'] = paper['pmid']
                e['source'] = 'title+abstract'
            
            all_entities.extend(entities)
            
            # Add entities to paper
            paper_with_ents = paper.copy()
            paper_with_ents['entities'] = entities
            papers_with_entities.append(paper_with_ents)
        
        # Calculate statistics
        entity_counts = Counter(e['type'] for e in all_entities)
        unique_by_type = defaultdict(set)
        for e in all_entities:
            unique_by_type[e['type']].add(e['text'])
        
        # Get top entities
        entity_freq = Counter(e['text'] for e in all_entities)
        top_entities = entity_freq.most_common(20)
        
        return {
            'query': actual_query,
            'papers': papers_with_entities,
            'total_papers': len(papers_with_entities),
            'entities': all_entities,
            'total_entities': len(all_entities),
            'entity_counts': dict(entity_counts),
            'unique_entities': {k: list(v) for k, v in unique_by_type.items()},
            'top_entities': top_entities,
        }
    
    def extract_relations(
        self,
        papers_data: Dict
    ) -> List[Dict]:
        """
        Extract potential relations between entities
        (Co-occurrence based)
        
        Args:
            papers_data: Output from search_and_extract
        
        Returns:
            List of relation dicts
        """
        relations = []
        
        for paper in papers_data['papers']:
            entities = paper.get('entities', [])
            
            # Group by type
            by_type = defaultdict(list)
            for e in entities:
                by_type[e['type']].append(e)
            
            # Gene-Disease relations
            for gene in by_type.get('GENE', []):
                for disease in by_type.get('DISEASE', []):
                    relations.append({
                        'source': gene['text'],
                        'source_type': 'GENE',
                        'target': disease['text'],
                        'target_type': 'DISEASE',
                        'relation': 'ASSOCIATED_WITH',
                        'pmid': paper['pmid'],
                        'evidence': f"Co-occurrence in PMID:{paper['pmid']}"
                    })
            
            # Drug-Disease relations
            for drug in by_type.get('DRUG', []):
                for disease in by_type.get('DISEASE', []):
                    relations.append({
                        'source': drug['text'],
                        'source_type': 'DRUG',
                        'target': disease['text'],
                        'target_type': 'DISEASE',
                        'relation': 'TREATS',
                        'pmid': paper['pmid'],
                        'evidence': f"Co-occurrence in PMID:{paper['pmid']}"
                    })
            
            # Gene-Gene relations
            genes = by_type.get('GENE', [])
            for i, g1 in enumerate(genes):
                for g2 in genes[i+1:]:
                    relations.append({
                        'source': g1['text'],
                        'source_type': 'GENE',
                        'target': g2['text'],
                        'target_type': 'GENE',
                        'relation': 'INTERACTS',
                        'pmid': paper['pmid'],
                        'evidence': f"Co-occurrence in PMID:{paper['pmid']}"
                    })
        
        return relations
    
    def get_entity_summary(
        self,
        papers_data: Dict
    ) -> str:
        """
        Generate text summary of extracted entities
        
        Args:
            papers_data: Output from search_and_extract
        
        Returns:
            Summary text
        """
        lines = [
            f"Literature Mining Summary",
            f"=" * 50,
            f"Query: {papers_data['query']}",
            f"Papers analyzed: {papers_data['total_papers']}",
            f"Total entities extracted: {papers_data['total_entities']}",
            f"",
            f"Entity counts by type:"
        ]
        
        for etype, count in papers_data['entity_counts'].items():
            unique_count = len(papers_data['unique_entities'].get(etype, []))
            lines.append(f"  {etype}: {count} mentions ({unique_count} unique)")
        
        lines.append(f"\nTop entities:")
        for entity, count in papers_data['top_entities'][:10]:
            lines.append(f"  {entity}: {count} mentions")
        
        return "\n".join(lines)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_entities_from_text(text: str, use_biobert: bool = False) -> List[Dict]:
    """
    Quick helper to extract entities from any text
    
    Args:
        text: Input text
        use_biobert: Use BioBERT model
    
    Returns:
        List of entity dicts
    """
    if use_biobert and TRANSFORMERS_AVAILABLE:
        try:
            ner = BioBERTNER()
        except:
            ner = RuleBasedNER()
    else:
        ner = RuleBasedNER()
    
    return ner.extract_entities(text)


def highlight_entities(text: str, entities: List[Dict]) -> str:
    """
    Create HTML with highlighted entities
    
    Args:
        text: Original text
        entities: List of entity dicts
    
    Returns:
        HTML string with highlighted entities
    """
    # Sort entities by position (reverse for replacement)
    entities_sorted = sorted(entities, key=lambda x: x['start'], reverse=True)
    
    # Color map
    colors = {
        'GENE': '#FFE5E5',      # Light red
        'DRUG': '#E5F3FF',      # Light blue
        'DISEASE': '#E5FFE5',   # Light green
        'MUTATION': '#FFF5E5',  # Light orange
        'PATHWAY': '#F5E5FF',   # Light purple
        'BIOMARKER': '#E5FFFF', # Light cyan
    }
    
    result = text
    for ent in entities_sorted:
        color = colors.get(ent['type'], '#F0F0F0')
        replacement = (
            f'<span style="background-color: {color}; '
            f'padding: 2px 4px; border-radius: 3px;" '
            f'title="{ent["type"]}">{ent["text"]}</span>'
        )
        result = result[:ent['start']] + replacement + result[ent['end']:]
    
    return result


def get_gene_disease_associations(papers_data: Dict) -> List[Tuple[str, str, int]]:
    """
    Extract gene-disease associations with counts
    
    Args:
        papers_data: Output from search_and_extract
    
    Returns:
        List of (gene, disease, count) tuples
    """
    associations = Counter()
    
    for paper in papers_data['papers']:
        entities = paper.get('entities', [])
        
        genes = [e['text'] for e in entities if e['type'] == 'GENE']
        diseases = [e['text'] for e in entities if e['type'] == 'DISEASE']
        
        for gene in genes:
            for disease in diseases:
                associations[(gene, disease)] += 1
    
    return [(g, d, c) for (g, d), c in associations.most_common()]


# Re-export base functions
__all__ = [
    # NER Classes
    'RuleBasedNER',
    'BioBERTNER',
    'LiteratureMiner',
    
    # Helper functions
    'extract_entities_from_text',
    'highlight_entities',
    'get_gene_disease_associations',
    
    # Base literature functions
    'fetch_pubmed',
    'PREDEFINED_QUERIES',
    'QUERY_OPTIONS',
    'get_query',
    
    # Constants
    'KNOWN_GENES',
    'KNOWN_DRUGS',
    'KNOWN_DISEASES',
    'TRANSFORMERS_AVAILABLE',
]


if __name__ == "__main__":
    print("Literature Mining with NER")
    print("=" * 50)
    
    # Test rule-based NER
    ner = RuleBasedNER()
    
    test_text = """
    BRCA1 and BRCA2 mutations are associated with increased risk of breast cancer.
    Tamoxifen is commonly used for treating ER-positive breast cancer patients.
    The PI3K/AKT signaling pathway plays a crucial role in cancer progression.
    HER2-positive tumors may respond to Trastuzumab therapy.
    """
    
    entities = ner.extract_entities(test_text)
    
    print(f"\nExtracted {len(entities)} entities:")
    for e in entities:
        print(f"  {e['type']}: {e['text']}")
    
    # Test with LiteratureMiner
    print("\nTesting LiteratureMiner...")
    miner = LiteratureMiner(use_biobert=False)
    
    results = miner.search_and_extract(
        "BRCA1/TP53/HER2 Biomarkers",  # Using predefined query
        max_results=3
    )
    
    print(miner.get_entity_summary(results))
    
    print("\n✅ Literature NER module working!")
