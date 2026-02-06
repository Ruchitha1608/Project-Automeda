"""
Clinical Trial Matcher Module
Match patient profiles to relevant breast cancer clinical trials

Uses ClinicalTrials.gov API to find matching trials based on:
- Molecular subtype
- Biomarkers and mutations
- Disease stage
- Prior treatments
"""

import requests
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import re


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class ClinicalTrial:
    nct_id: str
    title: str
    status: str
    phase: str
    conditions: List[str]
    interventions: List[str]
    eligibility: str
    brief_summary: str
    sponsor: str
    locations: List[str]
    start_date: Optional[str]
    completion_date: Optional[str]
    url: str
    match_score: float = 0.0
    match_reasons: List[str] = None


# ============================================================
# TRIAL DATABASE (Curated examples + API)
# ============================================================

# Pre-curated high-relevance trials (updated periodically)
CURATED_TRIALS = [
    {
        'nct_id': 'NCT04191135',
        'title': 'Trastuzumab Deruxtecan in HER2-Low Breast Cancer (DESTINY-Breast04)',
        'status': 'Active, not recruiting',
        'phase': 'Phase 3',
        'conditions': ['HER2-low Breast Cancer', 'Metastatic Breast Cancer'],
        'interventions': ['Trastuzumab Deruxtecan (T-DXd)'],
        'eligibility': 'HER2-low (IHC 1+ or IHC 2+/ISH-), hormone receptor positive or negative',
        'brief_summary': 'Study of trastuzumab deruxtecan vs chemotherapy in HER2-low metastatic breast cancer',
        'sponsor': 'Daiichi Sankyo',
        'locations': ['Multiple US and International Sites'],
        'biomarkers': ['HER2-low', 'HR+', 'HR-'],
        'subtypes': ['Luminal A', 'Luminal B', 'Basal-like (TNBC)'],
    },
    {
        'nct_id': 'NCT03901339',
        'title': 'Pembrolizumab + Chemotherapy for TNBC (KEYNOTE-522)',
        'status': 'Active, not recruiting',
        'phase': 'Phase 3',
        'conditions': ['Triple Negative Breast Cancer', 'Early Breast Cancer'],
        'interventions': ['Pembrolizumab', 'Chemotherapy'],
        'eligibility': 'Previously untreated, locally advanced, TNBC',
        'brief_summary': 'Neoadjuvant pembrolizumab plus chemotherapy followed by adjuvant pembrolizumab',
        'sponsor': 'Merck',
        'locations': ['Multiple US and International Sites'],
        'biomarkers': ['PD-L1', 'TNBC'],
        'subtypes': ['Basal-like (TNBC)'],
    },
    {
        'nct_id': 'NCT02437318',
        'title': 'Olaparib in BRCA-Mutated Breast Cancer (OlympiA)',
        'status': 'Active, not recruiting',
        'phase': 'Phase 3',
        'conditions': ['BRCA-Mutated Breast Cancer', 'HER2-Negative Breast Cancer'],
        'interventions': ['Olaparib'],
        'eligibility': 'Germline BRCA1/2 mutation, HER2-negative, high-risk early breast cancer',
        'brief_summary': 'Adjuvant olaparib after local treatment and neoadjuvant/adjuvant chemotherapy',
        'sponsor': 'AstraZeneca',
        'locations': ['Multiple US and International Sites'],
        'biomarkers': ['BRCA1', 'BRCA2', 'HER2-'],
        'subtypes': ['Basal-like (TNBC)', 'Luminal A', 'Luminal B'],
    },
    {
        'nct_id': 'NCT03155997',
        'title': 'CDK4/6 Inhibitor in HR+ Early Breast Cancer (monarchE)',
        'status': 'Active, not recruiting',
        'phase': 'Phase 3',
        'conditions': ['HR+ HER2- Breast Cancer', 'High Risk Early Breast Cancer'],
        'interventions': ['Abemaciclib', 'Endocrine Therapy'],
        'eligibility': 'HR+/HER2-, high-risk early breast cancer, node-positive',
        'brief_summary': 'Adjuvant abemaciclib combined with endocrine therapy',
        'sponsor': 'Eli Lilly',
        'locations': ['Multiple US and International Sites'],
        'biomarkers': ['HR+', 'HER2-', 'High Ki67'],
        'subtypes': ['Luminal A', 'Luminal B'],
    },
    {
        'nct_id': 'NCT04379596',
        'title': 'Sacituzumab Govitecan for HR+/HER2- MBC (TROPiCS-02)',
        'status': 'Active, not recruiting',
        'phase': 'Phase 3',
        'conditions': ['HR+ HER2- Metastatic Breast Cancer'],
        'interventions': ['Sacituzumab Govitecan'],
        'eligibility': 'HR+/HER2- MBC, prior endocrine, CDK4/6i, and 2-4 lines chemo',
        'brief_summary': 'Antibody-drug conjugate targeting Trop-2 in pretreated HR+/HER2- MBC',
        'sponsor': 'Gilead Sciences',
        'locations': ['Multiple US and International Sites'],
        'biomarkers': ['HR+', 'HER2-', 'TROP2'],
        'subtypes': ['Luminal A', 'Luminal B'],
    },
    {
        'nct_id': 'NCT03197935',
        'title': 'Alpelisib in PIK3CA-Mutated HR+ BC (SOLAR-1)',
        'status': 'Active, not recruiting',
        'phase': 'Phase 3',
        'conditions': ['PIK3CA-Mutated Breast Cancer', 'HR+ HER2- Breast Cancer'],
        'interventions': ['Alpelisib', 'Fulvestrant'],
        'eligibility': 'PIK3CA mutation, HR+/HER2-, postmenopausal, after prior AI',
        'brief_summary': 'PI3K inhibitor with fulvestrant in PIK3CA-mutated advanced breast cancer',
        'sponsor': 'Novartis',
        'locations': ['Multiple US and International Sites'],
        'biomarkers': ['PIK3CA', 'HR+', 'HER2-'],
        'subtypes': ['Luminal A', 'Luminal B'],
    },
    {
        'nct_id': 'NCT04585958',
        'title': 'Datopotamab Deruxtecan in TNBC (TROPION-Breast01)',
        'status': 'Recruiting',
        'phase': 'Phase 3',
        'conditions': ['Triple Negative Breast Cancer', 'Metastatic'],
        'interventions': ['Datopotamab Deruxtecan'],
        'eligibility': 'Locally recurrent inoperable or metastatic TNBC, 1-2 prior lines',
        'brief_summary': 'Trop2-directed ADC vs chemotherapy in previously treated TNBC',
        'sponsor': 'Daiichi Sankyo / AstraZeneca',
        'locations': ['Multiple Sites - Actively Recruiting'],
        'biomarkers': ['TNBC', 'TROP2'],
        'subtypes': ['Basal-like (TNBC)'],
    },
    {
        'nct_id': 'NCT05307705',
        'title': 'AKT Inhibitor Capivasertib in HR+/HER2- BC (CAPItello-290)',
        'status': 'Recruiting',
        'phase': 'Phase 3',
        'conditions': ['HR+ HER2- Breast Cancer', 'Advanced/Metastatic'],
        'interventions': ['Capivasertib', 'Fulvestrant'],
        'eligibility': 'HR+/HER2-, locally advanced/metastatic, after prior therapy',
        'brief_summary': 'AKT inhibitor combination in advanced HR+/HER2- breast cancer',
        'sponsor': 'AstraZeneca',
        'locations': ['Multiple Sites - Actively Recruiting'],
        'biomarkers': ['HR+', 'HER2-', 'AKT', 'PTEN'],
        'subtypes': ['Luminal A', 'Luminal B'],
    },
    {
        'nct_id': 'NCT05629585',
        'title': 'Next-Gen ADC in HER2+ Breast Cancer',
        'status': 'Recruiting',
        'phase': 'Phase 2',
        'conditions': ['HER2-Positive Breast Cancer', 'Brain Metastases'],
        'interventions': ['Novel HER2-ADC'],
        'eligibility': 'HER2+, CNS metastases, after prior trastuzumab',
        'brief_summary': 'Novel antibody-drug conjugate with CNS penetration',
        'sponsor': 'Multiple Sponsors',
        'locations': ['Select Academic Centers'],
        'biomarkers': ['HER2+', 'ERBB2'],
        'subtypes': ['HER2-enriched'],
    },
    {
        'nct_id': 'NCT05514054',
        'title': 'Bispecific Antibody in Advanced Breast Cancer',
        'status': 'Recruiting',
        'phase': 'Phase 1/2',
        'conditions': ['Advanced Breast Cancer', 'Solid Tumors'],
        'interventions': ['HER2xCD3 Bispecific'],
        'eligibility': 'HER2-expressing solid tumors, including breast',
        'brief_summary': 'Novel T-cell engaging bispecific antibody immunotherapy',
        'sponsor': 'Multiple Sponsors',
        'locations': ['Phase 1 Centers'],
        'biomarkers': ['HER2', 'CD3'],
        'subtypes': ['HER2-enriched', 'Luminal B'],
    },
]


# ============================================================
# CLINICAL TRIALS API
# ============================================================

def search_clinicaltrials_gov(query: str, max_results: int = 10) -> List[Dict]:
    """
    Search ClinicalTrials.gov API for breast cancer trials
    
    Args:
        query: Search query string
        max_results: Maximum number of results
    
    Returns:
        List of trial dictionaries
    """
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    
    params = {
        'query.term': f'breast cancer AND {query}',
        'filter.overallStatus': 'RECRUITING,ACTIVE_NOT_RECRUITING',
        'pageSize': max_results,
        'fields': 'NCTId,BriefTitle,OverallStatus,Phase,Condition,InterventionName,BriefSummary,LeadSponsorName,LocationCity,StartDate,CompletionDate,EligibilityCriteria',
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            trials = []
            
            for study in data.get('studies', []):
                protocol = study.get('protocolSection', {})
                id_module = protocol.get('identificationModule', {})
                status_module = protocol.get('statusModule', {})
                design_module = protocol.get('designModule', {})
                conditions_module = protocol.get('conditionsModule', {})
                interventions_module = protocol.get('armsInterventionsModule', {})
                desc_module = protocol.get('descriptionModule', {})
                sponsor_module = protocol.get('sponsorCollaboratorsModule', {})
                eligibility_module = protocol.get('eligibilityModule', {})
                locations_module = protocol.get('contactsLocationsModule', {})
                
                # Extract interventions
                interventions = []
                for intervention in interventions_module.get('interventions', []):
                    interventions.append(intervention.get('name', ''))
                    
                # Extract locations
                locations = []
                for loc in locations_module.get('locations', [])[:5]:
                    city = loc.get('city', '')
                    country = loc.get('country', '')
                    if city:
                        locations.append(f"{city}, {country}")
                
                trial = {
                    'nct_id': id_module.get('nctId', ''),
                    'title': id_module.get('briefTitle', ''),
                    'status': status_module.get('overallStatus', ''),
                    'phase': ','.join(design_module.get('phases', [])),
                    'conditions': conditions_module.get('conditions', []),
                    'interventions': interventions,
                    'eligibility': eligibility_module.get('eligibilityCriteria', '')[:500],
                    'brief_summary': desc_module.get('briefSummary', '')[:500],
                    'sponsor': sponsor_module.get('leadSponsor', {}).get('name', ''),
                    'locations': locations if locations else ['Check ClinicalTrials.gov'],
                    'start_date': status_module.get('startDateStruct', {}).get('date', ''),
                    'completion_date': status_module.get('completionDateStruct', {}).get('date', ''),
                }
                trials.append(trial)
                
            return trials
        else:
            print(f"API request failed: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error searching ClinicalTrials.gov: {e}")
        return []


# ============================================================
# TRIAL MATCHER
# ============================================================

class ClinicalTrialMatcher:
    """
    Match patient profiles to relevant clinical trials
    """
    
    def __init__(self):
        self.curated_trials = CURATED_TRIALS
        
    def match(self,
              subtype: str,
              biomarkers: Dict[str, float],
              stage: Optional[str] = None,
              brca_status: Optional[str] = None,
              her2_status: Optional[str] = None,
              er_status: Optional[str] = None,
              pr_status: Optional[str] = None,
              prior_treatments: Optional[List[str]] = None,
              include_api_search: bool = True) -> List[ClinicalTrial]:
        """
        Match patient to relevant clinical trials
        
        Args:
            subtype: Molecular subtype
            biomarkers: Gene expression profile
            stage: Disease stage
            brca_status: BRCA mutation status
            her2_status: HER2 status
            er_status: ER status
            pr_status: PR status
            prior_treatments: List of prior treatments
            include_api_search: Whether to also search live API
        
        Returns:
            List of matched ClinicalTrial objects, sorted by relevance
        """
        matched_trials = []
        
        # Build patient profile
        profile = self._build_profile(
            subtype, biomarkers, stage, brca_status, 
            her2_status, er_status, pr_status
        )
        
        # Match against curated trials
        for trial_data in self.curated_trials:
            score, reasons = self._calculate_match_score(trial_data, profile)
            if score > 0:
                trial = ClinicalTrial(
                    nct_id=trial_data['nct_id'],
                    title=trial_data['title'],
                    status=trial_data['status'],
                    phase=trial_data['phase'],
                    conditions=trial_data['conditions'],
                    interventions=trial_data['interventions'],
                    eligibility=trial_data['eligibility'],
                    brief_summary=trial_data['brief_summary'],
                    sponsor=trial_data['sponsor'],
                    locations=trial_data['locations'],
                    start_date=trial_data.get('start_date'),
                    completion_date=trial_data.get('completion_date'),
                    url=f"https://clinicaltrials.gov/study/{trial_data['nct_id']}",
                    match_score=score,
                    match_reasons=reasons,
                )
                matched_trials.append(trial)
                
        # Optionally search live API
        if include_api_search:
            api_trials = self._search_api_for_profile(profile)
            for trial_dict in api_trials:
                # Check if already in curated
                if not any(t.nct_id == trial_dict['nct_id'] for t in matched_trials):
                    score, reasons = self._calculate_api_match_score(trial_dict, profile)
                    if score > 0.3:
                        trial = ClinicalTrial(
                            nct_id=trial_dict['nct_id'],
                            title=trial_dict['title'],
                            status=trial_dict['status'],
                            phase=trial_dict['phase'],
                            conditions=trial_dict.get('conditions', []),
                            interventions=trial_dict.get('interventions', []),
                            eligibility=trial_dict.get('eligibility', ''),
                            brief_summary=trial_dict.get('brief_summary', ''),
                            sponsor=trial_dict.get('sponsor', ''),
                            locations=trial_dict.get('locations', []),
                            start_date=trial_dict.get('start_date'),
                            completion_date=trial_dict.get('completion_date'),
                            url=f"https://clinicaltrials.gov/study/{trial_dict['nct_id']}",
                            match_score=score,
                            match_reasons=reasons,
                        )
                        matched_trials.append(trial)
        
        # Sort by match score
        matched_trials.sort(key=lambda x: x.match_score, reverse=True)
        
        return matched_trials
    
    def _build_profile(self, subtype, biomarkers, stage, brca_status,
                       her2_status, er_status, pr_status) -> Dict:
        """Build patient profile for matching"""
        profile = {
            'subtype': subtype,
            'biomarkers': biomarkers,
            'stage': stage,
            'brca_status': brca_status,
            'her2_status': her2_status,
            'er_status': er_status,
            'pr_status': pr_status,
            'keywords': [],
        }
        
        # Generate keywords for matching
        if subtype:
            profile['keywords'].append(subtype.lower())
            
        if 'TNBC' in subtype or 'Basal' in subtype:
            profile['keywords'].extend(['tnbc', 'triple negative', 'basal'])
        if 'Luminal' in subtype:
            profile['keywords'].extend(['luminal', 'hr positive', 'hormone receptor'])
        if 'HER2' in subtype:
            profile['keywords'].extend(['her2', 'erbb2', 'her2-positive'])
            
        if brca_status == 'mutated':
            profile['keywords'].extend(['brca', 'brca1', 'brca2', 'parp'])
        if her2_status == 'positive':
            profile['keywords'].extend(['her2+', 'her2-positive'])
        elif her2_status == 'negative':
            profile['keywords'].extend(['her2-', 'her2-negative'])
        if er_status == 'positive':
            profile['keywords'].extend(['er+', 'er-positive'])
        if pr_status == 'positive':
            profile['keywords'].extend(['pr+', 'pr-positive'])
            
        # Check biomarkers for keywords
        for gene in biomarkers.keys():
            if gene.upper() in ['BRCA1', 'BRCA2', 'PIK3CA', 'PTEN', 'ESR1', 'ERBB2']:
                profile['keywords'].append(gene.lower())
                
        return profile
    
    def _calculate_match_score(self, trial: Dict, profile: Dict) -> Tuple[float, List[str]]:
        """Calculate match score for curated trial"""
        score = 0.0
        reasons = []
        
        # Check subtype match
        trial_subtypes = trial.get('subtypes', [])
        if profile['subtype'] in trial_subtypes:
            score += 0.4
            reasons.append(f"Matches your subtype: {profile['subtype']}")
        elif any(s.lower() in profile['subtype'].lower() for s in trial_subtypes):
            score += 0.3
            reasons.append("Subtype partially matches")
            
        # Check biomarker match
        trial_biomarkers = trial.get('biomarkers', [])
        for bm in trial_biomarkers:
            bm_lower = bm.lower()
            if profile['brca_status'] == 'mutated' and 'brca' in bm_lower:
                score += 0.3
                reasons.append("BRCA mutation - eligible for PARP inhibitor trials")
            if profile['her2_status'] == 'positive' and 'her2+' in bm_lower:
                score += 0.2
                reasons.append("HER2+ status matches")
            if profile['her2_status'] == 'negative' and 'her2-' in bm_lower:
                score += 0.15
                reasons.append("HER2- status matches")
            if profile['er_status'] == 'positive' and 'hr+' in bm_lower:
                score += 0.15
                reasons.append("HR+ status matches")
            if 'tnbc' in bm_lower and ('TNBC' in profile['subtype'] or 'Basal' in profile['subtype']):
                score += 0.25
                reasons.append("Triple-negative status matches")
                
            # Check if biomarker in patient's profile
            for gene in profile['biomarkers'].keys():
                if gene.lower() == bm_lower or gene.upper() == bm.upper():
                    score += 0.1
                    reasons.append(f"Biomarker {gene} relevant to trial")
                    
        # Boost for recruiting trials
        if 'Recruiting' in trial.get('status', ''):
            score += 0.1
            reasons.append("Trial is actively recruiting")
            
        return min(score, 1.0), reasons
    
    def _calculate_api_match_score(self, trial: Dict, profile: Dict) -> Tuple[float, List[str]]:
        """Calculate match score for API trial"""
        score = 0.0
        reasons = []
        
        # Check title and conditions for keyword matches
        title_lower = trial.get('title', '').lower()
        conditions_lower = ' '.join(trial.get('conditions', [])).lower()
        summary_lower = trial.get('brief_summary', '').lower()
        
        text_to_search = f"{title_lower} {conditions_lower} {summary_lower}"
        
        for keyword in profile['keywords']:
            if keyword in text_to_search:
                score += 0.15
                if len(reasons) < 3:
                    reasons.append(f"Matches: {keyword}")
                    
        # Boost Phase 3 trials
        if 'Phase 3' in trial.get('phase', '') or 'PHASE3' in trial.get('phase', ''):
            score += 0.1
            reasons.append("Phase 3 trial (larger, pivotal)")
            
        if 'Recruiting' in trial.get('status', ''):
            score += 0.1
            reasons.append("Actively recruiting")
            
        return min(score, 1.0), reasons
    
    def _search_api_for_profile(self, profile: Dict) -> List[Dict]:
        """Search API based on patient profile"""
        queries = []
        
        # Build targeted queries
        if 'Basal' in profile['subtype'] or 'TNBC' in profile['subtype']:
            queries.append('triple negative')
        elif 'HER2' in profile['subtype']:
            queries.append('HER2 positive')
        elif 'Luminal' in profile['subtype']:
            queries.append('hormone receptor positive')
            
        if profile['brca_status'] == 'mutated':
            queries.append('BRCA mutation')
            
        # Search and combine results
        all_trials = []
        for query in queries[:2]:  # Limit API calls
            trials = search_clinicaltrials_gov(query, max_results=5)
            all_trials.extend(trials)
            
        # Deduplicate
        seen = set()
        unique = []
        for trial in all_trials:
            if trial['nct_id'] not in seen:
                seen.add(trial['nct_id'])
                unique.append(trial)
                
        return unique
    
    def get_trial_details(self, nct_id: str) -> Optional[Dict]:
        """Get detailed information about a specific trial"""
        # First check curated
        for trial in self.curated_trials:
            if trial['nct_id'] == nct_id:
                return trial
                
        # Then try API
        trials = search_clinicaltrials_gov(nct_id, max_results=1)
        return trials[0] if trials else None
    
    def generate_report(self, matched_trials: List[ClinicalTrial], profile: Dict) -> str:
        """Generate a report of matched clinical trials"""
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║             CLINICAL TRIAL MATCHING REPORT                   ║
╚══════════════════════════════════════════════════════════════╝

PATIENT PROFILE:
  Molecular Subtype: {profile.get('subtype', 'Unknown')}
  HER2 Status: {profile.get('her2_status', 'Unknown')}
  ER Status: {profile.get('er_status', 'Unknown')}
  BRCA Status: {profile.get('brca_status', 'Unknown')}
  
═══════════════════════════════════════
MATCHED TRIALS: {len(matched_trials)} found
═══════════════════════════════════════

"""
        if not matched_trials:
            report += "No matching trials found. Consider broadening search criteria.\n"
        else:
            for i, trial in enumerate(matched_trials[:10], 1):
                match_pct = trial.match_score * 100
                status_emoji = "🟢" if "Recruiting" in trial.status else "🟡"
                
                report += f"""
{i}. {trial.title}
   {status_emoji} Status: {trial.status}
   📋 NCT ID: {trial.nct_id}
   📊 Phase: {trial.phase}
   🎯 Match Score: {match_pct:.0f}%
   
   Why this trial matches:
"""
                for reason in (trial.match_reasons or [])[:3]:
                    report += f"   • {reason}\n"
                    
                report += f"""
   Interventions: {', '.join(trial.interventions[:3])}
   Sponsor: {trial.sponsor}
   
   🔗 {trial.url}
   
   ─────────────────────────────────────
"""
                
        report += """
═══════════════════════════════════════
HOW TO PROCEED:
1. Discuss trial options with your oncologist
2. Visit ClinicalTrials.gov for full eligibility criteria
3. Contact trial sites for enrollment information

DISCLAIMER: Trial matching is based on general criteria.
Actual eligibility requires detailed medical review by
the trial investigators.
═══════════════════════════════════════
"""
        return report


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def find_matching_trials(subtype: str,
                         biomarkers: Dict[str, float],
                         **kwargs) -> List[ClinicalTrial]:
    """
    Find clinical trials matching patient profile
    
    Args:
        subtype: Molecular subtype
        biomarkers: Expression profile
        **kwargs: Additional clinical factors
    
    Returns:
        List of matched ClinicalTrial objects
    """
    matcher = ClinicalTrialMatcher()
    return matcher.match(subtype, biomarkers, **kwargs)


def get_curated_trials() -> List[Dict]:
    """Get list of curated high-relevance trials"""
    return CURATED_TRIALS
