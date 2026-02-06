"""
PDF Report Generator Module
Generate professional clinical reports combining all analyses

Creates comprehensive diagnostic reports including:
- Patient summary
- Imaging analysis results
- Genomics/biomarker analysis
- Molecular subtype classification
- Treatment recommendations
- Relevant literature
- Clinical trial matches
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import base64
import io

# Try to import PDF libraries
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, PageBreak, HRFlowable
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Also support HTML report generation
from html import escape


# ============================================================
# REPORT DATA STRUCTURES
# ============================================================

class DiagnosticReport:
    """Container for all diagnostic data"""
    
    def __init__(self):
        self.report_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.generated_at = datetime.now()
        
        # Patient info
        self.patient_id = None
        self.patient_age = None
        self.patient_notes = None
        
        # Analysis results
        self.imaging_result = None
        self.imaging_confidence = None
        self.imaging_heatmap = None
        
        self.omics_result = None
        self.omics_confidence = None
        self.biomarkers = []
        
        self.molecular_subtype = None
        self.subtype_probabilities = {}
        self.subtype_analysis = {}
        
        self.treatment_recommendations = {}
        
        self.clinical_trials = []
        
        self.literature_papers = []
        self.entities = []
        
        # Integrated results
        self.risk_level = None
        self.risk_score = None
        self.summary_text = None


# ============================================================
# HTML REPORT GENERATOR
# ============================================================

def generate_html_report(report: DiagnosticReport) -> str:
    """
    Generate an HTML report that can be printed or converted to PDF
    
    Args:
        report: DiagnosticReport object with all data
    
    Returns:
        HTML string
    """
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Breast Cancer Diagnostic Report - {report.report_id}</title>
    <style>
        @page {{
            margin: 0.75in;
        }}
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #1e3a5f, #2d5a87);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 28px;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .section {{
            background: #fff;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .section-title {{
            color: #1e3a5f;
            border-bottom: 2px solid #2d5a87;
            padding-bottom: 10px;
            margin-bottom: 15px;
            font-size: 20px;
        }}
        .risk-high {{
            background: #ffebee;
            border-left: 4px solid #c62828;
            color: #c62828;
        }}
        .risk-moderate {{
            background: #fff3e0;
            border-left: 4px solid #f57c00;
            color: #e65100;
        }}
        .risk-low {{
            background: #e8f5e9;
            border-left: 4px solid #2e7d32;
            color: #2e7d32;
        }}
        .result-box {{
            padding: 15px;
            border-radius: 6px;
            margin: 10px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        th {{
            background: #f5f5f5;
            font-weight: 600;
        }}
        .biomarker-bar {{
            background: #e0e0e0;
            height: 20px;
            border-radius: 10px;
            overflow: hidden;
        }}
        .biomarker-fill {{
            background: linear-gradient(90deg, #4CAF50, #2196F3);
            height: 100%;
            border-radius: 10px;
        }}
        .subtype-box {{
            text-align: center;
            padding: 20px;
            border-radius: 8px;
            margin: 10px 0;
        }}
        .treatment-item {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 6px;
            margin: 10px 0;
            border-left: 3px solid #2196F3;
        }}
        .trial-card {{
            background: #fafafa;
            padding: 15px;
            border-radius: 6px;
            margin: 10px 0;
            border: 1px solid #e0e0e0;
        }}
        .trial-status {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }}
        .status-recruiting {{
            background: #c8e6c9;
            color: #2e7d32;
        }}
        .status-active {{
            background: #fff9c4;
            color: #f57f17;
        }}
        .paper-item {{
            padding: 10px;
            border-bottom: 1px solid #eee;
        }}
        .footer {{
            text-align: center;
            color: #666;
            font-size: 12px;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
        }}
        .disclaimer {{
            background: #fff3e0;
            padding: 15px;
            border-radius: 6px;
            font-size: 13px;
            color: #e65100;
            margin-top: 20px;
        }}
        .metric {{
            display: inline-block;
            text-align: center;
            padding: 15px;
            margin: 5px;
            min-width: 120px;
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
            color: #1e3a5f;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
        }}
        @media print {{
            body {{ print-color-adjust: exact; -webkit-print-color-adjust: exact; }}
            .section {{ page-break-inside: avoid; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧬 Breast Cancer Diagnostic Report</h1>
        <p>AI-Powered Multimodal Analysis</p>
        <p style="font-size: 14px;">Report ID: {report.report_id} | Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M')}</p>
    </div>
"""
    
    # Executive Summary
    risk_class = 'risk-low' if report.risk_level == 'LOW' else ('risk-high' if report.risk_level == 'HIGH' else 'risk-moderate')
    html += f"""
    <div class="section">
        <h2 class="section-title">📋 Executive Summary</h2>
        <div class="result-box {risk_class}">
            <h3 style="margin-top: 0;">Overall Risk Assessment: {report.risk_level or 'PENDING'}</h3>
            <p>{escape(report.summary_text or 'Analysis pending. Complete all diagnostic steps.')}</p>
        </div>
        
        <div style="text-align: center; margin-top: 20px;">
            <div class="metric">
                <div class="metric-value">{report.imaging_result or '--'}</div>
                <div class="metric-label">Imaging Result</div>
            </div>
            <div class="metric">
                <div class="metric-value">{report.omics_result or '--'}</div>
                <div class="metric-label">Omics Result</div>
            </div>
            <div class="metric">
                <div class="metric-value">{report.molecular_subtype or '--'}</div>
                <div class="metric-label">Molecular Subtype</div>
            </div>
        </div>
    </div>
"""
    
    # Imaging Analysis
    if report.imaging_result:
        conf_pct = (report.imaging_confidence or 0) * 100
        result_class = 'risk-high' if report.imaging_result == 'Malignant' else 'risk-low'
        html += f"""
    <div class="section">
        <h2 class="section-title">🔬 Histopathology Imaging Analysis</h2>
        <div class="result-box {result_class}">
            <strong>Classification:</strong> {report.imaging_result}<br>
            <strong>Confidence:</strong> {conf_pct:.1f}%
        </div>
        <p><em>Analysis performed using ResNet50 deep learning model with GradCAM explainability.</em></p>
    </div>
"""
    
    # Omics Analysis
    if report.biomarkers:
        html += """
    <div class="section">
        <h2 class="section-title">🧬 Genomics & Biomarker Analysis</h2>
"""
        if report.omics_result:
            result_class = 'risk-high' if report.omics_result == 'Cancer' else 'risk-low'
            conf_pct = (report.omics_confidence or 0) * 100
            html += f"""
        <div class="result-box {result_class}">
            <strong>Classification:</strong> {report.omics_result}<br>
            <strong>Confidence:</strong> {conf_pct:.1f}%
        </div>
"""
        
        html += """
        <h4>Top Predictive Biomarkers</h4>
        <table>
            <tr><th>Gene</th><th>Importance</th><th>Score</th></tr>
"""
        for gene, imp in report.biomarkers[:10]:
            bar_width = min(imp * 100 / 0.3, 100)  # Scale to max ~0.3
            html += f"""
            <tr>
                <td><strong>{escape(gene)}</strong></td>
                <td>
                    <div class="biomarker-bar">
                        <div class="biomarker-fill" style="width: {bar_width}%"></div>
                    </div>
                </td>
                <td>{imp:.4f}</td>
            </tr>
"""
        html += """
        </table>
    </div>
"""
    
    # Molecular Subtype
    if report.molecular_subtype:
        subtype_colors = {
            'Luminal A': '#4CAF50',
            'Luminal B': '#8BC34A',
            'HER2-enriched': '#FF9800',
            'Basal-like (TNBC)': '#F44336',
            'Normal-like': '#9E9E9E',
        }
        color = subtype_colors.get(report.molecular_subtype, '#666')
        
        html += f"""
    <div class="section">
        <h2 class="section-title">🎯 Molecular Subtype Classification</h2>
        <div class="subtype-box" style="background: {color}20; border: 2px solid {color};">
            <h2 style="color: {color}; margin: 0;">{report.molecular_subtype}</h2>
        </div>
        
        <h4>Subtype Probability Distribution</h4>
        <table>
            <tr><th>Subtype</th><th>Probability</th></tr>
"""
        for subtype, prob in sorted(report.subtype_probabilities.items(), key=lambda x: x[1], reverse=True):
            bar_width = prob * 100
            html += f"""
            <tr>
                <td>{subtype}</td>
                <td>
                    <div class="biomarker-bar">
                        <div class="biomarker-fill" style="width: {bar_width}%; background: {subtype_colors.get(subtype, '#666')}"></div>
                    </div>
                    {prob:.1%}
                </td>
            </tr>
"""
        html += """
        </table>
    </div>
"""
    
    # Treatment Recommendations
    if report.treatment_recommendations:
        html += """
    <div class="section">
        <h2 class="section-title">💊 Treatment Recommendations</h2>
"""
        if report.treatment_recommendations.get('rationale'):
            html += "<p><strong>Clinical Rationale:</strong></p><ul>"
            for r in report.treatment_recommendations['rationale'][:3]:
                html += f"<li>{escape(r)}</li>"
            html += "</ul>"
            
        if report.treatment_recommendations.get('primary'):
            html += "<h4>Primary Recommendations</h4>"
            for tx in report.treatment_recommendations['primary'][:5]:
                html += f"""
        <div class="treatment-item">
            <strong>{escape(str(tx.get('treatment', 'Unknown')))}</strong><br>
            <em>{escape(str(tx.get('rationale', '')))}</em>
        </div>
"""
        
        if report.treatment_recommendations.get('biomarker_tests_recommended'):
            html += "<h4>Recommended Additional Tests</h4><ul>"
            for test in report.treatment_recommendations['biomarker_tests_recommended'][:5]:
                html += f"<li>{escape(test)}</li>"
            html += "</ul>"
            
        html += """
    </div>
"""
    
    # Clinical Trials
    if report.clinical_trials:
        html += """
    <div class="section">
        <h2 class="section-title">🔬 Matched Clinical Trials</h2>
        <p>The following clinical trials match your molecular profile:</p>
"""
        for trial in report.clinical_trials[:5]:
            status_class = 'status-recruiting' if 'Recruiting' in trial.status else 'status-active'
            html += f"""
        <div class="trial-card">
            <span class="trial-status {status_class}">{escape(trial.status)}</span>
            <h4 style="margin: 10px 0 5px 0;">{escape(trial.title)}</h4>
            <p style="margin: 5px 0; color: #666;">
                <strong>NCT ID:</strong> {trial.nct_id} | 
                <strong>Phase:</strong> {trial.phase} |
                <strong>Match:</strong> {trial.match_score:.0%}
            </p>
            <p style="margin: 5px 0; font-size: 14px;">{escape(trial.brief_summary[:200])}...</p>
            <a href="{trial.url}" target="_blank" style="color: #2196F3;">View on ClinicalTrials.gov →</a>
        </div>
"""
        html += """
    </div>
"""
    
    # Literature
    if report.literature_papers:
        html += """
    <div class="section">
        <h2 class="section-title">📚 Relevant Literature</h2>
"""
        for paper in report.literature_papers[:5]:
            html += f"""
        <div class="paper-item">
            <strong>{escape(paper.get('title', 'Unknown Title'))}</strong><br>
            <span style="color: #666; font-size: 13px;">
                {escape(', '.join(paper.get('authors', [])[:3]))}
                {' et al.' if len(paper.get('authors', [])) > 3 else ''}
                | {escape(paper.get('journal', 'Unknown Journal'))}
            </span><br>
            <a href="https://pubmed.ncbi.nlm.nih.gov/{paper.get('pmid', '')}" target="_blank" style="color: #2196F3; font-size: 13px;">
                PMID: {paper.get('pmid', 'N/A')}
            </a>
        </div>
"""
        html += """
    </div>
"""
    
    # Disclaimer and Footer
    html += """
    <div class="disclaimer">
        <strong>⚠️ Important Disclaimer:</strong> This report is generated by an AI diagnostic system 
        for research and educational purposes. All findings should be reviewed and validated by 
        qualified healthcare professionals. Treatment decisions must be made by licensed oncologists 
        in consultation with the patient, considering individual clinical circumstances.
    </div>
    
    <div class="footer">
        <p><strong>Breast Cancer AI Diagnostic System</strong></p>
        <p>Multimodal Analysis: Literature Mining • Histopathology Imaging • Genomics</p>
        <p>© 2024 - AI-Powered Diagnostic Research Tool</p>
    </div>
</body>
</html>
"""
    return html


# ============================================================
# PDF REPORT GENERATOR (if reportlab available)
# ============================================================

def generate_pdf_report(report: DiagnosticReport) -> Optional[bytes]:
    """
    Generate a PDF report using ReportLab
    
    Args:
        report: DiagnosticReport object with all data
    
    Returns:
        PDF bytes or None if reportlab not available
    """
    if not REPORTLAB_AVAILABLE:
        return None
        
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, 
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=72)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Title2', 
                              parent=styles['Heading1'],
                              fontSize=24,
                              alignment=TA_CENTER,
                              spaceAfter=20))
    styles.add(ParagraphStyle(name='SubTitle',
                              parent=styles['Normal'],
                              fontSize=12,
                              alignment=TA_CENTER,
                              textColor=colors.grey))
    styles.add(ParagraphStyle(name='SectionHeader',
                              parent=styles['Heading2'],
                              fontSize=14,
                              textColor=colors.HexColor('#1e3a5f'),
                              spaceBefore=20,
                              spaceAfter=10))
    
    story = []
    
    # Header
    story.append(Paragraph("🧬 Breast Cancer Diagnostic Report", styles['Title2']))
    story.append(Paragraph("AI-Powered Multimodal Analysis", styles['SubTitle']))
    story.append(Paragraph(f"Report ID: {report.report_id} | Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M')}", 
                          styles['SubTitle']))
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#2d5a87')))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    story.append(Paragraph("📋 Executive Summary", styles['SectionHeader']))
    
    risk_color = colors.green if report.risk_level == 'LOW' else (colors.red if report.risk_level == 'HIGH' else colors.orange)
    
    summary_data = [
        ['Risk Level', report.risk_level or 'PENDING'],
        ['Imaging Result', report.imaging_result or '--'],
        ['Omics Result', report.omics_result or '--'],
        ['Molecular Subtype', report.molecular_subtype or '--'],
    ]
    
    summary_table = Table(summary_data, colWidths=[2*inch, 3*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f5f5f5')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#333333')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 10))
    
    if report.summary_text:
        story.append(Paragraph(report.summary_text, styles['Normal']))
    
    # Biomarkers section
    if report.biomarkers:
        story.append(Paragraph("🧬 Top Predictive Biomarkers", styles['SectionHeader']))
        
        bm_data = [['Gene', 'Importance Score']]
        for gene, imp in report.biomarkers[:8]:
            bm_data.append([gene, f"{imp:.4f}"])
            
        bm_table = Table(bm_data, colWidths=[2.5*inch, 2.5*inch])
        bm_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a5f')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')]),
        ]))
        story.append(bm_table)
    
    # Treatment recommendations
    if report.treatment_recommendations and report.treatment_recommendations.get('primary'):
        story.append(Paragraph("💊 Treatment Recommendations", styles['SectionHeader']))
        
        for tx in report.treatment_recommendations['primary'][:3]:
            treatment_name = str(tx.get('treatment', 'Unknown'))
            rationale = str(tx.get('rationale', ''))
            story.append(Paragraph(f"<b>• {treatment_name}</b>", styles['Normal']))
            if rationale:
                story.append(Paragraph(f"  {rationale}", styles['Normal']))
            story.append(Spacer(1, 5))
    
    # Clinical Trials
    if report.clinical_trials:
        story.append(Paragraph("🔬 Matched Clinical Trials", styles['SectionHeader']))
        
        for trial in report.clinical_trials[:3]:
            story.append(Paragraph(f"<b>{trial.nct_id}</b> - {trial.title[:80]}...", styles['Normal']))
            story.append(Paragraph(f"Phase: {trial.phase} | Match: {trial.match_score:.0%}", styles['Normal']))
            story.append(Spacer(1, 8))
    
    # Disclaimer
    story.append(Spacer(1, 30))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "<b>Disclaimer:</b> This AI-generated report is for research purposes only. "
        "All findings must be reviewed by qualified healthcare professionals.",
        ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=9, textColor=colors.grey)
    ))
    
    # Build PDF
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_report(
    imaging_result: Optional[str] = None,
    imaging_confidence: Optional[float] = None,
    omics_result: Optional[str] = None,
    omics_confidence: Optional[float] = None,
    biomarkers: Optional[List] = None,
    molecular_subtype: Optional[str] = None,
    subtype_probabilities: Optional[Dict] = None,
    treatment_recommendations: Optional[Dict] = None,
    clinical_trials: Optional[List] = None,
    literature_papers: Optional[List] = None,
    risk_level: Optional[str] = None,
    summary_text: Optional[str] = None,
    **kwargs
) -> DiagnosticReport:
    """
    Create a DiagnosticReport from analysis results
    
    Args:
        All analysis results as keyword arguments
    
    Returns:
        DiagnosticReport object
    """
    report = DiagnosticReport()
    
    report.imaging_result = imaging_result
    report.imaging_confidence = imaging_confidence
    report.omics_result = omics_result
    report.omics_confidence = omics_confidence
    report.biomarkers = biomarkers or []
    report.molecular_subtype = molecular_subtype
    report.subtype_probabilities = subtype_probabilities or {}
    report.treatment_recommendations = treatment_recommendations or {}
    report.clinical_trials = clinical_trials or []
    report.literature_papers = literature_papers or []
    report.risk_level = risk_level
    report.summary_text = summary_text
    
    # Add any extra kwargs
    for key, value in kwargs.items():
        if hasattr(report, key):
            setattr(report, key, value)
            
    return report


def export_report(report: DiagnosticReport, format: str = 'html') -> Any:
    """
    Export report to specified format
    
    Args:
        report: DiagnosticReport object
        format: 'html' or 'pdf'
    
    Returns:
        HTML string or PDF bytes
    """
    if format.lower() == 'pdf':
        pdf = generate_pdf_report(report)
        if pdf is None:
            # Fall back to HTML if reportlab not available
            return generate_html_report(report)
        return pdf
    else:
        return generate_html_report(report)


def is_pdf_available() -> bool:
    """Check if PDF generation is available"""
    return REPORTLAB_AVAILABLE
