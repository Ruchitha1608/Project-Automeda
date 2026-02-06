"""
Treatment Recommendation Engine
AI-powered breast cancer therapy recommendations based on molecular profile

Provides evidence-based treatment suggestions considering:
- Molecular subtype
- Biomarker expression
- Clinical factors
- Latest guidelines (NCCN, ASCO, ESMO)
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


# ============================================================
# TREATMENT DATABASE
# ============================================================

class TherapyClass(Enum):
    HORMONE = "Hormone Therapy"
    CHEMOTHERAPY = "Chemotherapy"
    TARGETED = "Targeted Therapy"
    IMMUNOTHERAPY = "Immunotherapy"
    RADIATION = "Radiation Therapy"
    SURGERY = "Surgery"


@dataclass
class Treatment:
    name: str
    drug_class: TherapyClass
    mechanism: str
    indication: str
    evidence_level: str  # 1A, 1B, 2A, 2B, 3
    side_effects: List[str]
    monitoring: List[str]
    contraindications: List[str]


# Comprehensive treatment database
TREATMENTS = {
    # === HORMONE THERAPY ===
    'tamoxifen': Treatment(
        name="Tamoxifen",
        drug_class=TherapyClass.HORMONE,
        mechanism="Selective Estrogen Receptor Modulator (SERM)",
        indication="ER+ breast cancer, pre/post-menopausal",
        evidence_level="1A",
        side_effects=["Hot flashes", "Thromboembolic events", "Endometrial cancer risk"],
        monitoring=["Annual gynecologic exam", "Lipid profile", "Liver function"],
        contraindications=["History of DVT/PE", "Pregnancy", "Concurrent warfarin"]
    ),
    'anastrozole': Treatment(
        name="Anastrozole (Arimidex)",
        drug_class=TherapyClass.HORMONE,
        mechanism="Aromatase Inhibitor (AI)",
        indication="ER+ breast cancer, postmenopausal only",
        evidence_level="1A",
        side_effects=["Joint pain/stiffness", "Bone loss", "Hot flashes"],
        monitoring=["Bone density (DEXA)", "Lipid profile", "Joint symptoms"],
        contraindications=["Premenopausal status", "Severe osteoporosis"]
    ),
    'letrozole': Treatment(
        name="Letrozole (Femara)",
        drug_class=TherapyClass.HORMONE,
        mechanism="Aromatase Inhibitor (AI)",
        indication="ER+ breast cancer, postmenopausal, extended adjuvant",
        evidence_level="1A",
        side_effects=["Arthralgia", "Hot flashes", "Fatigue", "Bone loss"],
        monitoring=["Bone density", "Lipid profile"],
        contraindications=["Premenopausal status"]
    ),
    'fulvestrant': Treatment(
        name="Fulvestrant (Faslodex)",
        drug_class=TherapyClass.HORMONE,
        mechanism="Selective Estrogen Receptor Degrader (SERD)",
        indication="ER+ metastatic breast cancer, after AI progression",
        evidence_level="1A",
        side_effects=["Injection site reactions", "Nausea", "Bone pain"],
        monitoring=["Disease progression imaging"],
        contraindications=["Severe hepatic impairment"]
    ),
    
    # === TARGETED THERAPY ===
    'trastuzumab': Treatment(
        name="Trastuzumab (Herceptin)",
        drug_class=TherapyClass.TARGETED,
        mechanism="HER2 monoclonal antibody",
        indication="HER2+ breast cancer",
        evidence_level="1A",
        side_effects=["Cardiotoxicity", "Infusion reactions", "Diarrhea"],
        monitoring=["LVEF (echocardiogram) every 3 months", "Cardiac symptoms"],
        contraindications=["LVEF <50%", "Severe cardiac disease"]
    ),
    'pertuzumab': Treatment(
        name="Pertuzumab (Perjeta)",
        drug_class=TherapyClass.TARGETED,
        mechanism="HER2 dimerization inhibitor",
        indication="HER2+ breast cancer, combined with trastuzumab",
        evidence_level="1A",
        side_effects=["Diarrhea", "Rash", "Cardiotoxicity (with trastuzumab)"],
        monitoring=["LVEF monitoring", "Diarrhea management"],
        contraindications=["Cardiac dysfunction"]
    ),
    'trastuzumab_deruxtecan': Treatment(
        name="Trastuzumab Deruxtecan (Enhertu)",
        drug_class=TherapyClass.TARGETED,
        mechanism="HER2-directed antibody-drug conjugate",
        indication="HER2+ or HER2-low metastatic breast cancer",
        evidence_level="1A",
        side_effects=["ILD/pneumonitis", "Nausea", "Fatigue", "Neutropenia"],
        monitoring=["Pulmonary symptoms", "CBC", "LVEF"],
        contraindications=["ILD history"]
    ),
    'palbociclib': Treatment(
        name="Palbociclib (Ibrance)",
        drug_class=TherapyClass.TARGETED,
        mechanism="CDK4/6 inhibitor",
        indication="HR+/HER2- advanced breast cancer",
        evidence_level="1A",
        side_effects=["Neutropenia", "Fatigue", "Nausea", "Infections"],
        monitoring=["CBC every 2 weeks initially, then monthly"],
        contraindications=["Severe hepatic impairment"]
    ),
    'ribociclib': Treatment(
        name="Ribociclib (Kisqali)",
        drug_class=TherapyClass.TARGETED,
        mechanism="CDK4/6 inhibitor",
        indication="HR+/HER2- advanced breast cancer",
        evidence_level="1A",
        side_effects=["Neutropenia", "QT prolongation", "Hepatotoxicity"],
        monitoring=["ECG", "CBC", "LFTs"],
        contraindications=["Long QT syndrome", "Concurrent QT-prolonging drugs"]
    ),
    'olaparib': Treatment(
        name="Olaparib (Lynparza)",
        drug_class=TherapyClass.TARGETED,
        mechanism="PARP inhibitor",
        indication="BRCA-mutated HER2- breast cancer",
        evidence_level="1A",
        side_effects=["Anemia", "Fatigue", "Nausea", "MDS/AML risk"],
        monitoring=["CBC monthly", "MDS symptoms"],
        contraindications=["Myelodysplastic syndrome"]
    ),
    'alpelisib': Treatment(
        name="Alpelisib (Piqray)",
        drug_class=TherapyClass.TARGETED,
        mechanism="PI3K inhibitor",
        indication="PIK3CA-mutated HR+/HER2- breast cancer",
        evidence_level="1A",
        side_effects=["Hyperglycemia", "Rash", "Diarrhea", "GI effects"],
        monitoring=["Fasting glucose", "HbA1c", "Skin assessment"],
        contraindications=["Uncontrolled diabetes", "History of severe skin reactions"]
    ),
    
    # === IMMUNOTHERAPY ===
    'pembrolizumab': Treatment(
        name="Pembrolizumab (Keytruda)",
        drug_class=TherapyClass.IMMUNOTHERAPY,
        mechanism="PD-1 checkpoint inhibitor",
        indication="TNBC with PD-L1 CPS ≥10, early-stage high-risk TNBC",
        evidence_level="1A",
        side_effects=["Immune-mediated adverse events", "Fatigue", "Rash", "Colitis"],
        monitoring=["Thyroid function", "LFTs", "Immune-related AEs"],
        contraindications=["Active autoimmune disease", "Organ transplant"]
    ),
    
    # === CHEMOTHERAPY ===
    'doxorubicin': Treatment(
        name="Doxorubicin (Adriamycin)",
        drug_class=TherapyClass.CHEMOTHERAPY,
        mechanism="Anthracycline - DNA intercalation, topoisomerase II inhibition",
        indication="Breast cancer, adjuvant/neoadjuvant",
        evidence_level="1A",
        side_effects=["Cardiotoxicity (cumulative)", "Myelosuppression", "Alopecia", "Nausea"],
        monitoring=["LVEF at baseline and periodically", "CBC", "Cumulative dose"],
        contraindications=["LVEF <50%", "Prior anthracycline exposure at max dose"]
    ),
    'paclitaxel': Treatment(
        name="Paclitaxel (Taxol)",
        drug_class=TherapyClass.CHEMOTHERAPY,
        mechanism="Taxane - microtubule stabilization",
        indication="Breast cancer, adjuvant/neoadjuvant/metastatic",
        evidence_level="1A",
        side_effects=["Peripheral neuropathy", "Myelosuppression", "Alopecia", "Hypersensitivity"],
        monitoring=["CBC", "Neuropathy assessment", "Premedication for HSR"],
        contraindications=["Severe neuropathy", "Severe hypersensitivity history"]
    ),
    'capecitabine': Treatment(
        name="Capecitabine (Xeloda)",
        drug_class=TherapyClass.CHEMOTHERAPY,
        mechanism="Oral fluoropyrimidine prodrug",
        indication="Metastatic breast cancer, post-neoadjuvant for residual disease",
        evidence_level="1A",
        side_effects=["Hand-foot syndrome", "Diarrhea", "Nausea", "Myelosuppression"],
        monitoring=["CBC", "Renal function", "DPD deficiency screening"],
        contraindications=["DPD deficiency", "Severe renal impairment"]
    ),
}


# ============================================================
# TREATMENT REGIMENS
# ============================================================

REGIMENS = {
    'AC': {
        'name': 'AC (Doxorubicin + Cyclophosphamide)',
        'drugs': ['doxorubicin', 'cyclophosphamide'],
        'schedule': 'Every 2-3 weeks x 4 cycles',
        'indication': 'Adjuvant/neoadjuvant for early breast cancer',
    },
    'AC-T': {
        'name': 'AC-T (AC followed by Taxane)',
        'drugs': ['doxorubicin', 'cyclophosphamide', 'paclitaxel'],
        'schedule': 'AC q2-3wk x 4, then paclitaxel weekly x 12 or q2wk x 4',
        'indication': 'High-risk early breast cancer',
    },
    'TCH': {
        'name': 'TCH (Docetaxel + Carboplatin + Trastuzumab)',
        'drugs': ['docetaxel', 'carboplatin', 'trastuzumab'],
        'schedule': 'Every 3 weeks x 6 cycles',
        'indication': 'HER2+ breast cancer (cardiac-sparing option)',
    },
    'TCHP': {
        'name': 'TCHP (Docetaxel + Carboplatin + Trastuzumab + Pertuzumab)',
        'drugs': ['docetaxel', 'carboplatin', 'trastuzumab', 'pertuzumab'],
        'schedule': 'Every 3 weeks x 6 cycles',
        'indication': 'HER2+ breast cancer, standard of care',
    },
    'ddAC-THP': {
        'name': 'Dose-dense AC-THP',
        'drugs': ['doxorubicin', 'cyclophosphamide', 'paclitaxel', 'trastuzumab', 'pertuzumab'],
        'schedule': 'ddAC q2wk x 4, then THP q3wk x 4',
        'indication': 'High-risk HER2+ breast cancer',
    },
}


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

class TreatmentRecommender:
    """
    Evidence-based treatment recommendation engine
    Considers molecular profile, biomarkers, and clinical factors
    """
    
    def __init__(self):
        self.treatments = TREATMENTS
        self.regimens = REGIMENS
        
    def recommend(self, 
                  subtype: str,
                  biomarkers: Dict[str, float],
                  stage: Optional[str] = None,
                  menopausal_status: Optional[str] = None,
                  brca_status: Optional[str] = None,
                  pd_l1_status: Optional[str] = None,
                  pik3ca_status: Optional[str] = None,
                  prior_treatments: Optional[List[str]] = None) -> Dict:
        """
        Generate personalized treatment recommendations
        
        Args:
            subtype: Molecular subtype (from classifier)
            biomarkers: Gene expression profile
            stage: Cancer stage (I, II, III, IV)
            menopausal_status: 'pre' or 'post' menopausal
            brca_status: 'mutated' or 'wild-type'
            pd_l1_status: 'positive' or 'negative'
            pik3ca_status: 'mutated' or 'wild-type'
            prior_treatments: List of prior treatments
        
        Returns:
            Dict with recommendations, rationale, and alternatives
        """
        recommendations = {
            'primary': [],
            'alternatives': [],
            'clinical_trials_suggested': [],
            'biomarker_tests_recommended': [],
            'rationale': [],
            'monitoring': [],
            'warnings': [],
        }
        
        # Generate recommendations based on subtype
        if 'Luminal A' in subtype:
            self._recommend_luminal_a(recommendations, menopausal_status, stage, biomarkers)
        elif 'Luminal B' in subtype:
            self._recommend_luminal_b(recommendations, menopausal_status, stage, biomarkers)
        elif 'HER2' in subtype:
            self._recommend_her2(recommendations, stage, biomarkers)
        elif 'Basal' in subtype or 'TNBC' in subtype:
            self._recommend_tnbc(recommendations, stage, brca_status, pd_l1_status, biomarkers)
        else:
            self._recommend_default(recommendations, biomarkers)
            
        # Add biomarker-specific recommendations
        self._add_biomarker_specific(recommendations, biomarkers, brca_status, pik3ca_status)
        
        # Add recommended tests
        self._recommend_tests(recommendations, subtype, biomarkers)
        
        # Check for warnings based on biomarkers
        self._add_warnings(recommendations, biomarkers, prior_treatments)
        
        return recommendations
    
    def _recommend_luminal_a(self, rec: Dict, menopause: str, stage: str, biomarkers: Dict):
        """Recommendations for Luminal A subtype"""
        rec['rationale'].append(
            "Luminal A tumors are hormone receptor-positive with low proliferation. "
            "Endocrine therapy is the primary treatment; chemotherapy often not needed."
        )
        
        if menopause == 'pre':
            rec['primary'].append({
                'treatment': 'tamoxifen',
                'details': self.treatments['tamoxifen'],
                'duration': '5-10 years',
                'rationale': 'Standard for premenopausal ER+ breast cancer',
            })
            rec['alternatives'].append({
                'treatment': 'Ovarian suppression + AI',
                'rationale': 'Consider for high-risk premenopausal patients',
            })
        else:  # postmenopausal
            rec['primary'].append({
                'treatment': 'anastrozole',
                'details': self.treatments['anastrozole'],
                'duration': '5-10 years',
                'rationale': 'Aromatase inhibitors preferred in postmenopausal',
            })
            rec['alternatives'].append({
                'treatment': 'letrozole',
                'details': self.treatments['letrozole'],
                'rationale': 'Alternative AI option',
            })
            
        # Usually no chemo needed for Luminal A
        if stage and stage.upper() in ['I', 'II']:
            rec['rationale'].append(
                "For stage I-II Luminal A, chemotherapy typically NOT recommended. "
                "Consider Oncotype DX to confirm low recurrence risk."
            )
        
        rec['monitoring'].extend([
            "Annual mammogram and clinical breast exam",
            "Bone density monitoring (if on AI)",
            "Lipid panel annually",
        ])
        
    def _recommend_luminal_b(self, rec: Dict, menopause: str, stage: str, biomarkers: Dict):
        """Recommendations for Luminal B subtype"""
        rec['rationale'].append(
            "Luminal B tumors have higher proliferation than Luminal A. "
            "Combined endocrine therapy and chemotherapy often beneficial."
        )
        
        # Endocrine therapy
        if menopause == 'pre':
            rec['primary'].append({
                'treatment': 'tamoxifen',
                'details': self.treatments['tamoxifen'],
                'duration': '5-10 years',
                'rationale': 'Standard endocrine backbone',
            })
        else:
            rec['primary'].append({
                'treatment': 'letrozole',
                'details': self.treatments['letrozole'],
                'duration': '5-10 years',
                'rationale': 'AI preferred in postmenopausal',
            })
            
        # CDK4/6 inhibitors for advanced disease
        if stage and stage.upper() in ['III', 'IV']:
            rec['primary'].append({
                'treatment': 'palbociclib',
                'details': self.treatments['palbociclib'],
                'rationale': 'CDK4/6 inhibitor improves PFS in advanced HR+/HER2- disease',
            })
            
        # Chemotherapy often needed
        rec['primary'].append({
            'treatment': 'AC-T regimen',
            'details': REGIMENS['AC-T'],
            'rationale': 'Chemotherapy recommended for Luminal B due to higher proliferation',
        })
        
        rec['monitoring'].extend([
            "CBC during chemotherapy",
            "LVEF if receiving anthracyclines",
            "Genomic testing to guide decisions",
        ])
        
    def _recommend_her2(self, rec: Dict, stage: str, biomarkers: Dict):
        """Recommendations for HER2-enriched subtype"""
        rec['rationale'].append(
            "HER2-positive breast cancer responds well to HER2-targeted therapies. "
            "Dual HER2 blockade (trastuzumab + pertuzumab) is standard of care."
        )
        
        # HER2-targeted therapy
        rec['primary'].append({
            'treatment': 'trastuzumab',
            'details': self.treatments['trastuzumab'],
            'duration': '1 year (adjuvant)',
            'rationale': 'Foundation of HER2+ treatment',
        })
        rec['primary'].append({
            'treatment': 'pertuzumab',
            'details': self.treatments['pertuzumab'],
            'rationale': 'Add to trastuzumab for dual HER2 blockade',
        })
        
        # Chemotherapy backbone
        rec['primary'].append({
            'treatment': 'TCHP regimen',
            'details': REGIMENS['TCHP'],
            'rationale': 'Standard regimen for HER2+ breast cancer',
        })
        
        # For metastatic/resistant
        if stage and stage.upper() == 'IV':
            rec['alternatives'].append({
                'treatment': 'trastuzumab_deruxtecan',
                'details': self.treatments['trastuzumab_deruxtecan'],
                'rationale': 'Antibody-drug conjugate for metastatic HER2+ disease',
            })
            
        rec['monitoring'].extend([
            "LVEF monitoring every 3 months during HER2 therapy",
            "Watch for infusion reactions",
            "Cardiac symptoms assessment",
        ])
        
    def _recommend_tnbc(self, rec: Dict, stage: str, brca_status: str, pd_l1_status: str, biomarkers: Dict):
        """Recommendations for Triple-Negative Breast Cancer"""
        rec['rationale'].append(
            "Triple-negative breast cancer lacks ER, PR, and HER2. "
            "Chemotherapy is the backbone; newer targeted options available for specific biomarkers."
        )
        
        # Chemotherapy
        rec['primary'].append({
            'treatment': 'AC-T regimen',
            'details': REGIMENS['AC-T'],
            'rationale': 'Anthracycline/taxane backbone for TNBC',
        })
        
        # Immunotherapy for PD-L1+
        if pd_l1_status == 'positive':
            rec['primary'].append({
                'treatment': 'pembrolizumab',
                'details': self.treatments['pembrolizumab'],
                'rationale': 'Add immunotherapy for PD-L1+ TNBC (CPS ≥10)',
            })
        else:
            rec['biomarker_tests_recommended'].append(
                "PD-L1 testing (CPS score) - immunotherapy may benefit if positive"
            )
            
        # PARP inhibitors for BRCA-mutated
        if brca_status == 'mutated':
            rec['primary'].append({
                'treatment': 'olaparib',
                'details': self.treatments['olaparib'],
                'rationale': 'PARP inhibitor indicated for gBRCA-mutated HER2- breast cancer',
            })
        else:
            rec['biomarker_tests_recommended'].append(
                "Germline BRCA1/2 testing - PARP inhibitors effective if mutated"
            )
            
        # Post-neoadjuvant capecitabine
        rec['alternatives'].append({
            'treatment': 'capecitabine',
            'details': self.treatments['capecitabine'],
            'rationale': 'Consider for residual disease after neoadjuvant therapy',
        })
        
        rec['monitoring'].extend([
            "CBC during chemotherapy",
            "Immune-related adverse events if on immunotherapy",
            "Close follow-up due to higher recurrence risk",
        ])
        
    def _recommend_default(self, rec: Dict, biomarkers: Dict):
        """Default recommendations when subtype unclear"""
        rec['rationale'].append(
            "Subtype not clearly determined. Recommend comprehensive biomarker testing "
            "and multidisciplinary tumor board review."
        )
        rec['biomarker_tests_recommended'].extend([
            "ER/PR IHC",
            "HER2 IHC and/or FISH",
            "Ki67 proliferation index",
            "Oncotype DX or MammaPrint if HR+/HER2-",
        ])
        
    def _add_biomarker_specific(self, rec: Dict, biomarkers: Dict, brca_status: str, pik3ca_status: str):
        """Add recommendations based on specific biomarkers"""
        
        # Check for high proliferation markers
        for gene, val in biomarkers.items():
            if gene.upper() == 'MKI67' and val > 0.20:
                rec['rationale'].append(
                    f"High Ki67 (MKI67) expression detected - suggests higher proliferation, "
                    "may benefit from more aggressive therapy."
                )
                
        # PIK3CA mutation
        if pik3ca_status == 'mutated':
            rec['alternatives'].append({
                'treatment': 'alpelisib',
                'details': self.treatments['alpelisib'],
                'rationale': 'PI3K inhibitor for PIK3CA-mutated HR+/HER2- disease',
            })
        else:
            # Check if PIK3CA testing recommended
            for gene in biomarkers:
                if 'PIK3' in gene.upper():
                    rec['biomarker_tests_recommended'].append(
                        "PIK3CA mutation testing - targeted therapy available if mutated"
                    )
                    break
                    
    def _recommend_tests(self, rec: Dict, subtype: str, biomarkers: Dict):
        """Recommend additional tests based on profile"""
        
        if 'Luminal' in subtype:
            rec['biomarker_tests_recommended'].extend([
                "Oncotype DX Recurrence Score (if node-negative or 1-3 nodes)",
                "MammaPrint/PAM50 for risk stratification",
            ])
            
        if 'TNBC' in subtype or 'Basal' in subtype:
            rec['biomarker_tests_recommended'].extend([
                "Germline genetic testing (BRCA1/2, PALB2)",
                "Tumor mutational burden (TMB)",
            ])
            
        # Suggest clinical trials
        rec['clinical_trials_suggested'].extend([
            "Consider clinical trial enrollment - novel agents in development",
            "Check ClinicalTrials.gov for biomarker-specific trials",
        ])
        
    def _add_warnings(self, rec: Dict, biomarkers: Dict, prior_treatments: Optional[List[str]]):
        """Add warnings based on profile and treatment history"""
        
        if prior_treatments:
            if 'doxorubicin' in prior_treatments or 'anthracycline' in str(prior_treatments).lower():
                rec['warnings'].append(
                    "⚠️ Prior anthracycline exposure - monitor cumulative dose, "
                    "consider cardiac-sparing regimens"
                )
                
        # Check for potential drug interactions/contraindications
        for gene, val in biomarkers.items():
            if gene.upper() == 'DPYD' and val < 0.1:
                rec['warnings'].append(
                    "⚠️ Low DPYD expression - screen for DPD deficiency before fluoropyrimidines "
                    "(capecitabine, 5-FU). Risk of severe toxicity."
                )
                
    def get_treatment_info(self, treatment_key: str) -> Optional[Treatment]:
        """Get detailed information about a treatment"""
        return self.treatments.get(treatment_key)
    
    def get_all_treatments(self) -> Dict[str, Treatment]:
        """Get all treatments"""
        return self.treatments
    
    def generate_report(self, recommendations: Dict, subtype: str) -> str:
        """Generate a text report of recommendations"""
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║             TREATMENT RECOMMENDATION REPORT                  ║
╚══════════════════════════════════════════════════════════════╝

MOLECULAR SUBTYPE: {subtype}

CLINICAL RATIONALE:
"""
        for r in recommendations['rationale']:
            report += f"  • {r}\n"
            
        report += "\n═══════════════════════════════════════\n"
        report += "PRIMARY TREATMENT RECOMMENDATIONS:\n"
        report += "═══════════════════════════════════════\n\n"
        
        for i, tx in enumerate(recommendations['primary'], 1):
            treatment = tx.get('treatment', 'Unknown')
            details = tx.get('details')
            report += f"{i}. {treatment.upper()}\n"
            if isinstance(details, Treatment):
                report += f"   Class: {details.drug_class.value}\n"
                report += f"   Mechanism: {details.mechanism}\n"
                report += f"   Evidence: Level {details.evidence_level}\n"
            elif isinstance(details, dict):
                report += f"   Regimen: {details.get('name', '')}\n"
                report += f"   Schedule: {details.get('schedule', '')}\n"
            if 'rationale' in tx:
                report += f"   Rationale: {tx['rationale']}\n"
            if 'duration' in tx:
                report += f"   Duration: {tx['duration']}\n"
            report += "\n"
            
        if recommendations['alternatives']:
            report += "\n═══════════════════════════════════════\n"
            report += "ALTERNATIVE OPTIONS:\n"
            report += "═══════════════════════════════════════\n\n"
            for tx in recommendations['alternatives']:
                report += f"  • {tx.get('treatment', 'Unknown')}: {tx.get('rationale', '')}\n"
                
        if recommendations['biomarker_tests_recommended']:
            report += "\n═══════════════════════════════════════\n"
            report += "RECOMMENDED ADDITIONAL TESTS:\n"
            report += "═══════════════════════════════════════\n\n"
            for test in recommendations['biomarker_tests_recommended']:
                report += f"  ✓ {test}\n"
                
        if recommendations['warnings']:
            report += "\n═══════════════════════════════════════\n"
            report += "WARNINGS & PRECAUTIONS:\n"
            report += "═══════════════════════════════════════\n\n"
            for warning in recommendations['warnings']:
                report += f"  {warning}\n"
                
        if recommendations['monitoring']:
            report += "\n═══════════════════════════════════════\n"
            report += "MONITORING RECOMMENDATIONS:\n"
            report += "═══════════════════════════════════════\n\n"
            for item in recommendations['monitoring']:
                report += f"  → {item}\n"
                
        report += """
═══════════════════════════════════════
DISCLAIMER: These recommendations are AI-generated suggestions 
based on general guidelines. All treatment decisions should be 
made by qualified oncologists in consultation with the patient,
considering individual clinical circumstances.
═══════════════════════════════════════
"""
        return report


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def get_treatment_recommendations(subtype: str,
                                   biomarkers: Dict[str, float],
                                   **kwargs) -> Dict:
    """
    Get personalized treatment recommendations
    
    Args:
        subtype: Molecular subtype
        biomarkers: Expression profile
        **kwargs: Additional clinical factors
    
    Returns:
        Dict with recommendations
    """
    recommender = TreatmentRecommender()
    return recommender.recommend(subtype, biomarkers, **kwargs)


def get_therapy_classes() -> List[str]:
    """Get list of therapy classes"""
    return [tc.value for tc in TherapyClass]
