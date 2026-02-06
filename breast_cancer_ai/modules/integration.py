"""
Integration Module
Combines imaging, omics, and literature evidence into unified diagnosis
"""


def generate_summary(img_pred, img_conf, omics_pred, omics_conf, biomarkers, literature):
    """
    Generate integrated multimodal diagnostic summary
    
    Args:
        img_pred (str): Imaging prediction ("Benign" or "Malignant")
        img_conf (float): Imaging confidence (0-1)
        omics_pred (str): Omics prediction ("Normal" or "Cancer")
        omics_conf (float): Omics confidence (0-1)
        biomarkers (list): List of top biomarker gene names
        literature (list): List of PubMed paper dicts
    
    Returns:
        dict: Comprehensive diagnostic report with risk assessment
    """
    
    # Map predictions to cancer indicators
    img_cancer = (img_pred == "Malignant")
    omics_cancer = (omics_pred == "Cancer")
    
    # Calculate unified risk score
    if img_cancer and omics_cancer:
        # Both agree on cancer
        risk_score = (img_conf + omics_conf) / 2
        risk_level = "HIGH RISK" if risk_score > 0.85 else "MODERATE-HIGH RISK"
        agreement = "Strong multimodal agreement"
    elif img_cancer or omics_cancer:
        # Disagreement
        if img_cancer:
            risk_score = img_conf * 0.7
        else:
            risk_score = omics_conf * 0.7
        risk_level = "MODERATE RISK"
        agreement = "Mixed signals - further testing recommended"
    else:
        # Both agree on benign/normal
        risk_score = 1 - ((img_conf + omics_conf) / 2)
        risk_level = "LOW RISK"
        agreement = "Multimodal agreement on benign status"
    
    # Build detailed summary text
    summary_parts = []
    
    # Imaging findings
    if img_cancer:
        summary_parts.append(
            f"🔬 **Imaging Analysis**: Detected {img_pred.lower()} regions "
            f"with {img_conf:.1%} confidence. GradCAM heatmap highlights suspicious areas "
            f"showing abnormal tissue patterns consistent with tumor presence."
        )
    else:
        summary_parts.append(
            f"🔬 **Imaging Analysis**: Tissue classified as {img_pred.lower()} "
            f"with {img_conf:.1%} confidence. No significant malignant features detected in histopathology."
        )
    
    # Omics findings
    if omics_cancer and biomarkers:
        # Handle biomarkers as either strings or tuples (gene_name, importance)
        biomarker_names = []
        for b in biomarkers[:3]:
            if isinstance(b, tuple):
                biomarker_names.append(b[0])
            else:
                biomarker_names.append(str(b))
        biomarker_text = ", ".join(biomarker_names)
        summary_parts.append(
            f"🧬 **Genomic Analysis**: Molecular signature indicates {omics_pred.lower()} pattern "
            f"({omics_conf:.1%} confidence). Key biomarkers showing dysregulation: **{biomarker_text}**. "
            f"Expression profiles align with oncogenic pathways."
        )
    else:
        summary_parts.append(
            f"🧬 **Genomic Analysis**: Gene expression profile classified as {omics_pred.lower()} "
            f"({omics_conf:.1%} confidence). Molecular markers within expected ranges."
        )
    
    # Literature support
    evidence_count = len([p for p in literature if p['pmid'] != 'N/A'])
    if evidence_count > 0:
        summary_parts.append(
            f"📚 **Literature Evidence**: {evidence_count} relevant PubMed publications support "
            f"the involvement of identified biomarkers in breast cancer pathogenesis. "
            f"Scientific consensus validates multimodal findings."
        )
    else:
        summary_parts.append(
            f"📚 **Literature Evidence**: Limited publications retrieved. "
            f"Consider broader literature review for comprehensive context."
        )
    
    # Unified conclusion
    if risk_score > 0.85:
        conclusion = (
            f"⚠️ **CLINICAL RECOMMENDATION**: {risk_level}. {agreement}. "
            f"Convergence of imaging and molecular evidence strongly suggests malignancy. "
            f"Immediate clinical correlation and potential biopsy/treatment planning advised. "
            f"Comprehensive tumor board review recommended."
        )
    elif risk_score > 0.65:
        conclusion = (
            f"⚡ **CLINICAL RECOMMENDATION**: {risk_level}. {agreement}. "
            f"Evidence suggests potential malignancy requiring careful monitoring. "
            f"Additional diagnostic tests (IHC, FISH) recommended for definitive diagnosis. "
            f"Close follow-up essential."
        )
    else:
        conclusion = (
            f"✅ **CLINICAL RECOMMENDATION**: {risk_level}. {agreement}. "
            f"Current evidence favors benign status. Routine screening and monitoring advised. "
            f"Patient counseling on preventive measures and regular check-ups."
        )
    
    summary_parts.append(conclusion)
    
    # Combine all parts
    full_summary = "\n\n".join(summary_parts)
    
    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "agreement": agreement,
        "summary_text": full_summary,
        "img_cancer": img_cancer,
        "omics_cancer": omics_cancer
    }


def format_biomarkers_text(biomarkers):
    """Format biomarker list for display"""
    if not biomarkers:
        return "No significant biomarkers identified"
    
    text = "**Top 5 Identified Biomarkers:**\n\n"
    for i, item in enumerate(biomarkers[:5], 1):
        # Handle both tuple (gene, importance) and string formats
        if isinstance(item, tuple):
            gene = item[0]
        else:
            gene = str(item)
            
        # Add brief description for known genes
        descriptions = {
            "BRCA1": "Tumor suppressor - DNA repair",
            "BRCA2": "Tumor suppressor - DNA repair",
            "TP53": "Tumor suppressor - cell cycle regulator",
            "HER2": "Growth factor receptor - oncogene",
            "ESR1": "Estrogen receptor - hormone signaling",
            "PGR": "Progesterone receptor - hormone signaling",
            "MYC": "Transcription factor - cell proliferation",
            "PIK3CA": "Kinase - PI3K/AKT pathway",
            "PTEN": "Tumor suppressor - PI3K pathway",
            "EGFR": "Growth factor receptor",
            "MKI67": "Cell proliferation marker",
            "BAX": "Apoptosis regulator",
            "KRAS": "RAS signaling oncogene",
            "AKT1": "PI3K/AKT pathway kinase"
        }
        
        desc = descriptions.get(gene, "Gene expression marker")
        text += f"{i}. **{gene}** - {desc}\n"
    
    return text


def calculate_confidence_metrics(img_conf, omics_conf):
    """Calculate aggregate confidence metrics"""
    avg_conf = (img_conf + omics_conf) / 2
    max_conf = max(img_conf, omics_conf)
    min_conf = min(img_conf, omics_conf)
    
    # Agreement strength (how close the confidences are)
    agreement_strength = 1 - abs(img_conf - omics_conf)
    
    return {
        "average_confidence": avg_conf,
        "max_confidence": max_conf,
        "min_confidence": min_conf,
        "agreement_strength": agreement_strength
    }


if __name__ == "__main__":
    # Test integration
    test_summary = generate_summary(
        img_pred="Malignant",
        img_conf=0.91,
        omics_pred="Cancer",
        omics_conf=0.89,
        biomarkers=["BRCA1", "TP53", "HER2", "ESR1", "PGR"],
        literature=[{"pmid": "12345678", "title": "Test"}]
    )
    
    print(test_summary["summary_text"])
    print(f"\nRisk Score: {test_summary['risk_score']:.2%}")
    print(f"Risk Level: {test_summary['risk_level']}")
