# Multimodal AI for Breast Cancer Detection - Module Package
__version__ = "1.0.0"

# Core modules
from .literature import (
    fetch_pubmed,
    PREDEFINED_QUERIES,
    QUERY_OPTIONS,
    get_query
)

from .imaging import predict_image
from .omics import analyze_omics, analyze_omics_file
from .integration import generate_summary

# Enhanced modules
from .literature_ner import (
    RuleBasedNER,
    LiteratureMiner,
    extract_entities_from_text,
    highlight_entities,
    get_gene_disease_associations,
    KNOWN_GENES,
    KNOWN_DRUGS,
    KNOWN_DISEASES
)

from .knowledge_graph import (
    KnowledgeGraph,
    create_graph_from_literature,
    visualize_biomarker_network,
    ENTITY_COLORS
)

from .explainability import (
    GradCAM,
    GradCAMPlusPlus,
    IntegratedGradients,
    FeatureImportance,
    ExplanationGenerator,
    create_explanation_figure
)

__all__ = [
    # Literature
    'fetch_pubmed',
    'PREDEFINED_QUERIES',
    'QUERY_OPTIONS',
    'get_query',
    
    # NER
    'RuleBasedNER',
    'LiteratureMiner',
    'extract_entities_from_text',
    'highlight_entities',
    'get_gene_disease_associations',
    'KNOWN_GENES',
    'KNOWN_DRUGS',
    'KNOWN_DISEASES',
    
    # Core analysis
    'predict_image',
    'analyze_omics',
    'generate_summary',
    
    # Knowledge Graph
    'KnowledgeGraph',
    'create_graph_from_literature',
    'visualize_biomarker_network',
    'ENTITY_COLORS',
    
    # Explainability
    'GradCAM',
    'GradCAMPlusPlus',
    'IntegratedGradients',
    'FeatureImportance',
    'ExplanationGenerator',
    'create_explanation_figure',
]
