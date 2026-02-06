"""
Molecular Subtype Classifier Module
PAM50-style breast cancer molecular subtyping based on biomarker expression

Subtypes:
- Luminal A: ER+/PR+, HER2-, low Ki67 (best prognosis)
- Luminal B: ER+/PR+, HER2±, high Ki67 (moderate prognosis)
- HER2-enriched: ER-/PR-, HER2+ (targeted therapy available)
- Basal-like/TNBC: ER-/PR-, HER2- (aggressive, limited therapy)
- Normal-like: Similar to normal breast tissue
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import json

# ============================================================
# PAM50 GENE SIGNATURES
# ============================================================

PAM50_GENES = {
    # Luminal markers (high in Luminal A/B)
    'luminal': ['ESR1', 'PGR', 'FOXA1', 'GATA3', 'XBP1', 'MYB', 'KRT8', 'KRT18'],
    
    # Proliferation markers (high in Luminal B, HER2+, Basal)
    'proliferation': ['MKI67', 'CCNB1', 'MYBL2', 'AURKA', 'BIRC5', 'CCNE1', 'UBE2C', 'CDC20'],
    
    # HER2 markers
    'her2': ['ERBB2', 'GRB7', 'EGFR'],
    
    # Basal markers (high in Basal-like/TNBC)
    'basal': ['KRT5', 'KRT14', 'KRT17', 'FOXC1', 'CDH3', 'SFRP1', 'ITGB4'],
    
    # Normal-like markers
    'normal': ['CD36', 'FABP4', 'ADIPOQ', 'GPD1', 'ADH1B'],
}


# ============================================================
# SUBTYPE DEFINITIONS
# ============================================================

SUBTYPE_INFO = {
    'Luminal A': {
        'color': '#4CAF50',  # Green
        'description': 'Most common (~40%). ER+/PR+, HER2-, low proliferation. Best prognosis.',
        'characteristics': [
            'Estrogen receptor positive',
            'Progesterone receptor positive', 
            'HER2 negative',
            'Low Ki67 proliferation index (<20%)',
            'Low grade tumors common',
        ],
        'treatment': [
            'Hormone therapy (Tamoxifen, Aromatase inhibitors)',
            'Usually no chemotherapy needed',
            'Excellent response to endocrine therapy',
        ],
        'prognosis': 'Excellent (5-year survival >95%)',
        'prevalence': '40-50%',
    },
    'Luminal B': {
        'color': '#8BC34A',  # Light green
        'description': 'ER+/PR+, HER2±, high proliferation. Moderate prognosis.',
        'characteristics': [
            'Estrogen receptor positive',
            'Progesterone receptor variable',
            'HER2 may be positive or negative',
            'High Ki67 proliferation index (≥20%)',
            'More aggressive than Luminal A',
        ],
        'treatment': [
            'Hormone therapy plus chemotherapy',
            'Anti-HER2 therapy if HER2+',
            'May benefit from CDK4/6 inhibitors',
        ],
        'prognosis': 'Good (5-year survival ~90%)',
        'prevalence': '15-20%',
    },
    'HER2-enriched': {
        'color': '#FF9800',  # Orange
        'description': 'ER-/PR-, HER2+. Aggressive but targetable with HER2 therapies.',
        'characteristics': [
            'Estrogen receptor negative',
            'Progesterone receptor negative',
            'HER2 overexpression/amplification',
            'High proliferation rate',
            'Often high grade at diagnosis',
        ],
        'treatment': [
            'HER2-targeted therapy (Trastuzumab, Pertuzumab)',
            'Often combined with chemotherapy',
            'Antibody-drug conjugates (T-DM1, T-DXd)',
        ],
        'prognosis': 'Improved with targeted therapy (5-year survival ~85%)',
        'prevalence': '15-20%',
    },
    'Basal-like (TNBC)': {
        'color': '#F44336',  # Red
        'description': 'Triple negative (ER-/PR-/HER2-). Most aggressive, limited therapy options.',
        'characteristics': [
            'Triple negative (ER-/PR-/HER2-)',
            'High proliferation rate',
            'Often BRCA1 mutation-associated',
            'Common in younger women',
            'Higher grade, larger tumors',
        ],
        'treatment': [
            'Chemotherapy (anthracyclines, taxanes)',
            'PARP inhibitors if BRCA mutated',
            'Immunotherapy (Pembrolizumab) for PD-L1+',
            'Emerging targeted therapies',
        ],
        'prognosis': 'Variable (5-year survival ~77%)',
        'prevalence': '10-15%',
    },
    'Normal-like': {
        'color': '#9E9E9E',  # Gray
        'description': 'Similar to normal breast tissue. May be sampling artifact.',
        'characteristics': [
            'Gene expression similar to normal breast',
            'Low proliferation markers',
            'May represent low tumor cellularity',
            'Controversial clinical significance',
        ],
        'treatment': [
            'Treatment based on clinical factors',
            'Often treated as Luminal A',
            'Consider re-biopsy for confirmation',
        ],
        'prognosis': 'Generally good',
        'prevalence': '<5%',
    },
}


# ============================================================
# CLASSIFIER
# ============================================================

class MolecularSubtypeClassifier:
    """
    PAM50-inspired molecular subtype classifier for breast cancer
    Uses gene expression patterns to classify into 5 subtypes
    """
    
    def __init__(self):
        self.subtype_info = SUBTYPE_INFO
        self.pam50_genes = PAM50_GENES
        
    def calculate_signature_scores(self, biomarkers: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate signature scores for each marker category
        
        Args:
            biomarkers: Dict of gene -> expression/importance value
        
        Returns:
            Dict of category -> average score
        """
        scores = {}
        
        for category, genes in self.pam50_genes.items():
            values = []
            for gene in genes:
                # Check if gene is in biomarkers (case-insensitive)
                for bm_gene, value in biomarkers.items():
                    if bm_gene.upper() == gene.upper():
                        values.append(value)
                        break
            
            if values:
                scores[category] = np.mean(values)
            else:
                scores[category] = 0.0
                
        return scores
    
    def classify(self, biomarkers: Dict[str, float], 
                 er_status: Optional[str] = None,
                 pr_status: Optional[str] = None,
                 her2_status: Optional[str] = None) -> Tuple[str, Dict[str, float], Dict]:
        """
        Classify breast cancer molecular subtype
        
        Args:
            biomarkers: Dict of gene -> expression value
            er_status: Optional clinical ER status ('positive'/'negative')
            pr_status: Optional clinical PR status ('positive'/'negative')
            her2_status: Optional clinical HER2 status ('positive'/'negative')
        
        Returns:
            Tuple of (subtype, confidence_scores, detailed_analysis)
        """
        # Calculate signature scores
        sig_scores = self.calculate_signature_scores(biomarkers)
        
        # Normalize scores
        total = sum(sig_scores.values()) + 1e-6
        norm_scores = {k: v/total for k, v in sig_scores.items()}
        
        # Calculate subtype probabilities using decision rules
        probabilities = self._calculate_probabilities(
            norm_scores, er_status, pr_status, her2_status, biomarkers
        )
        
        # Get predicted subtype
        predicted_subtype = max(probabilities, key=probabilities.get)
        
        # Create detailed analysis
        analysis = {
            'signature_scores': sig_scores,
            'normalized_scores': norm_scores,
            'probabilities': probabilities,
            'key_markers': self._identify_key_markers(biomarkers),
            'clinical_correlation': self._correlate_clinical(
                er_status, pr_status, her2_status
            ),
        }
        
        return predicted_subtype, probabilities, analysis
    
    def _calculate_probabilities(self, norm_scores: Dict[str, float],
                                  er_status: Optional[str],
                                  pr_status: Optional[str],
                                  her2_status: Optional[str],
                                  biomarkers: Dict[str, float]) -> Dict[str, float]:
        """Calculate subtype probabilities based on scores and clinical data"""
        
        probs = {subtype: 0.0 for subtype in SUBTYPE_INFO.keys()}
        
        luminal = norm_scores.get('luminal', 0)
        prolif = norm_scores.get('proliferation', 0)
        her2 = norm_scores.get('her2', 0)
        basal = norm_scores.get('basal', 0)
        normal = norm_scores.get('normal', 0)
        
        # Check Ki67 if available
        ki67_high = False
        for gene, val in biomarkers.items():
            if gene.upper() == 'MKI67' or gene.upper() == 'KI67':
                ki67_high = val > 0.15  # Threshold
                break
        
        # Rule-based classification with soft probabilities
        
        # Luminal A: High ER, Low proliferation
        if luminal > 0.2 and prolif < 0.25:
            probs['Luminal A'] = 0.7 + luminal * 0.3 - prolif * 0.2
        elif luminal > 0.15:
            probs['Luminal A'] = 0.4 + luminal * 0.2
            
        # Luminal B: High ER, High proliferation
        if luminal > 0.15 and prolif > 0.15:
            probs['Luminal B'] = 0.5 + prolif * 0.3 + luminal * 0.1
        if ki67_high and luminal > 0.1:
            probs['Luminal B'] += 0.2
            
        # HER2-enriched: High HER2, Low ER
        if her2 > 0.15 and luminal < 0.2:
            probs['HER2-enriched'] = 0.6 + her2 * 0.4
        elif her2 > 0.2:
            probs['HER2-enriched'] = 0.4 + her2 * 0.3
            
        # Basal-like: High basal, Low ER, Low HER2
        if basal > 0.15 and luminal < 0.15 and her2 < 0.15:
            probs['Basal-like (TNBC)'] = 0.7 + basal * 0.3
        elif basal > 0.2:
            probs['Basal-like (TNBC)'] = 0.5 + basal * 0.3
            
        # Normal-like: High normal signature
        if normal > 0.25 and prolif < 0.15:
            probs['Normal-like'] = 0.5 + normal * 0.5
            
        # Adjust based on clinical IHC if provided
        if er_status == 'negative' and pr_status == 'negative':
            probs['Luminal A'] *= 0.1
            probs['Luminal B'] *= 0.2
            if her2_status == 'negative':
                probs['Basal-like (TNBC)'] *= 1.5
            elif her2_status == 'positive':
                probs['HER2-enriched'] *= 1.5
                
        if her2_status == 'positive':
            probs['HER2-enriched'] *= 1.3
            probs['Basal-like (TNBC)'] *= 0.3
            
        # Normalize probabilities
        total = sum(probs.values()) + 1e-6
        probs = {k: min(v/total, 1.0) for k, v in probs.items()}
        
        # Ensure at least one subtype has reasonable probability
        if max(probs.values()) < 0.2:
            probs['Luminal A'] = 0.3  # Default to most common
            
        return probs
    
    def _identify_key_markers(self, biomarkers: Dict[str, float]) -> List[Dict]:
        """Identify key diagnostic markers from biomarker list"""
        key_markers = []
        
        important_genes = {
            'ESR1': ('Estrogen Receptor', 'luminal'),
            'PGR': ('Progesterone Receptor', 'luminal'),
            'ERBB2': ('HER2', 'her2'),
            'MKI67': ('Ki67 Proliferation', 'proliferation'),
            'EGFR': ('EGFR', 'basal'),
            'KRT5': ('Cytokeratin 5', 'basal'),
            'KRT14': ('Cytokeratin 14', 'basal'),
            'BRCA1': ('BRCA1 Tumor Suppressor', 'dna_repair'),
            'BRCA2': ('BRCA2 Tumor Suppressor', 'dna_repair'),
            'TP53': ('p53 Tumor Suppressor', 'cell_cycle'),
        }
        
        for gene, value in biomarkers.items():
            gene_upper = gene.upper()
            if gene_upper in important_genes:
                name, category = important_genes[gene_upper]
                key_markers.append({
                    'gene': gene,
                    'name': name,
                    'value': value,
                    'category': category,
                    'interpretation': self._interpret_marker(gene_upper, value)
                })
                
        # Sort by importance value
        key_markers.sort(key=lambda x: x['value'], reverse=True)
        return key_markers[:10]
    
    def _interpret_marker(self, gene: str, value: float) -> str:
        """Interpret individual marker value"""
        if gene in ['ESR1', 'PGR']:
            if value > 0.2:
                return 'High expression - hormone receptor positive'
            elif value > 0.1:
                return 'Moderate expression'
            else:
                return 'Low expression - likely hormone receptor negative'
        elif gene == 'ERBB2':
            if value > 0.2:
                return 'High expression - HER2 positive'
            else:
                return 'Normal expression - HER2 negative'
        elif gene == 'MKI67':
            if value > 0.2:
                return 'High proliferation index (>20%)'
            else:
                return 'Low proliferation index'
        elif gene in ['KRT5', 'KRT14', 'EGFR']:
            if value > 0.15:
                return 'High expression - basal phenotype marker'
            else:
                return 'Normal expression'
        else:
            if value > 0.2:
                return 'High expression'
            elif value > 0.1:
                return 'Moderate expression'
            else:
                return 'Low expression'
    
    def _correlate_clinical(self, er_status: Optional[str],
                           pr_status: Optional[str],
                           her2_status: Optional[str]) -> Dict:
        """Correlate with clinical IHC results"""
        correlation = {
            'er': er_status,
            'pr': pr_status,
            'her2': her2_status,
            'clinical_subtype': None,
            'agreement': None,
        }
        
        if er_status and pr_status and her2_status:
            # Determine clinical subtype
            if er_status == 'positive' or pr_status == 'positive':
                if her2_status == 'positive':
                    correlation['clinical_subtype'] = 'Hormone Receptor+/HER2+'
                else:
                    correlation['clinical_subtype'] = 'Hormone Receptor+/HER2-'
            else:
                if her2_status == 'positive':
                    correlation['clinical_subtype'] = 'HER2+'
                else:
                    correlation['clinical_subtype'] = 'Triple Negative'
                    
        return correlation
    
    def get_subtype_info(self, subtype: str) -> Dict:
        """Get detailed information about a subtype"""
        return self.subtype_info.get(subtype, {})
    
    def generate_report(self, subtype: str, probabilities: Dict[str, float],
                       analysis: Dict) -> str:
        """Generate a text report for the classification"""
        info = self.subtype_info.get(subtype, {})
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║           MOLECULAR SUBTYPE CLASSIFICATION REPORT            ║
╚══════════════════════════════════════════════════════════════╝

PREDICTED SUBTYPE: {subtype}
CONFIDENCE: {probabilities.get(subtype, 0):.1%}

SUBTYPE DESCRIPTION:
{info.get('description', 'N/A')}

CHARACTERISTICS:
"""
        for char in info.get('characteristics', []):
            report += f"  • {char}\n"
            
        report += f"""
RECOMMENDED TREATMENT APPROACHES:
"""
        for tx in info.get('treatment', []):
            report += f"  → {tx}\n"
            
        report += f"""
PROGNOSIS: {info.get('prognosis', 'N/A')}
PREVALENCE: {info.get('prevalence', 'N/A')}

PROBABILITY DISTRIBUTION:
"""
        for st, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            bar = '█' * int(prob * 20)
            report += f"  {st:20s} {prob:6.1%} {bar}\n"
            
        if analysis.get('key_markers'):
            report += "\nKEY DIAGNOSTIC MARKERS:\n"
            for marker in analysis['key_markers'][:5]:
                report += f"  • {marker['gene']} ({marker['name']}): {marker['value']:.3f}\n"
                report += f"    {marker['interpretation']}\n"
        
        return report


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def classify_subtype(biomarkers: Dict[str, float], 
                     er_status: Optional[str] = None,
                     pr_status: Optional[str] = None,
                     her2_status: Optional[str] = None) -> Tuple[str, Dict[str, float], Dict]:
    """
    Convenience function to classify molecular subtype
    
    Args:
        biomarkers: Dict of gene -> expression value (from omics analysis)
        er_status: Optional ER status ('positive'/'negative')
        pr_status: Optional PR status ('positive'/'negative')
        her2_status: Optional HER2 status ('positive'/'negative')
    
    Returns:
        Tuple of (predicted_subtype, probabilities, analysis_dict)
    """
    classifier = MolecularSubtypeClassifier()
    return classifier.classify(biomarkers, er_status, pr_status, her2_status)


def get_subtype_color(subtype: str) -> str:
    """Get the display color for a subtype"""
    return SUBTYPE_INFO.get(subtype, {}).get('color', '#9E9E9E')


def get_all_subtypes() -> List[str]:
    """Get list of all subtypes"""
    return list(SUBTYPE_INFO.keys())
