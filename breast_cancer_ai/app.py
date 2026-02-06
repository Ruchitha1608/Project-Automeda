"""
🏥 Breast Cancer AI - Multimodal Diagnostic System
Clean, Step-by-Step Interface
"""

import streamlit as st
from PIL import Image
import pandas as pd
import sys
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent))

from modules import literature, imaging, omics, integration
from modules.literature_ner import LiteratureMiner, highlight_entities, get_gene_disease_associations, TRANSFORMERS_AVAILABLE
from modules.knowledge_graph import KnowledgeGraph
from database import EntitiesDatabase, get_database

# NEW: Advanced analysis modules
from modules.subtype_classifier import MolecularSubtypeClassifier, classify_subtype, get_subtype_color, SUBTYPE_INFO
from modules.treatment_recommender import TreatmentRecommender, get_treatment_recommendations
from modules.clinical_trials import ClinicalTrialMatcher, find_matching_trials
from modules.report_generator import create_report, export_report, DiagnosticReport, is_pdf_available

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Breast Cancer AI",
    page_icon="🧬",
    layout="wide"
)

# Simple CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #1e3a5f, #2d5a87);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .result-cancer {
        background: #ffebee;
        border-left: 4px solid #c62828;
    }
    .result-normal {
        background: #e8f5e9;
        border-left: 4px solid #2e7d32;
    }
    .biomarker-item {
        padding: 0.5rem;
        background: #f5f5f5;
        margin: 0.3rem 0;
        border-radius: 4px;
    }
    .paper-card {
        padding: 1rem;
        background: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .metric-card {
        text-align: center;
        padding: 1rem;
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .risk-high { color: #c62828; font-weight: bold; }
    .risk-moderate { color: #f57c00; font-weight: bold; }
    .risk-low { color: #2e7d32; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="main-header">
    <h1>🧬 Breast Cancer AI Diagnostic System</h1>
    <p>Multimodal Analysis: Literature • Imaging • Genomics</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================

if 'literature_results' not in st.session_state:
    st.session_state.literature_results = None
if 'literature_ner' not in st.session_state:
    st.session_state.literature_ner = None
if 'imaging_results' not in st.session_state:
    st.session_state.imaging_results = None
if 'omics_results' not in st.session_state:
    st.session_state.omics_results = None
if 'use_biobert' not in st.session_state:
    st.session_state.use_biobert = False
if 'db' not in st.session_state:
    st.session_state.db = get_database()
# NEW: Session state for advanced features
if 'subtype_result' not in st.session_state:
    st.session_state.subtype_result = None
if 'treatment_recs' not in st.session_state:
    st.session_state.treatment_recs = None
if 'matched_trials' not in st.session_state:
    st.session_state.matched_trials = None

# ============================================================
# TABS - Step by Step
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📚 1. Literature Mining",
    "🔬 2. Imaging Analysis", 
    "🧬 3. Omics Analysis",
    "📊 4. Integrated Results",
    "🧪 5. Advanced Insights",
    "🗄️ 6. Database History"
])

# ============================================================
# TAB 1: LITERATURE MINING
# ============================================================

with tab1:
    st.header("📚 Literature Mining")
    st.write("Search PubMed for relevant breast cancer research papers.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Query selection
        query_options = literature.QUERY_OPTIONS
        selected_query = st.selectbox(
            "Select a predefined query:",
            options=["Custom query..."] + query_options
        )
        
        if selected_query == "Custom query...":
            query = st.text_input("Enter your custom PubMed query:", 
                                  value="breast cancer BRCA1")
        else:
            query = literature.get_query(selected_query)
            st.info(f"**Query:** {query}")
    
    with col2:
        max_papers = st.slider("Number of papers", 3, 20, 5)
    
    # BioBERT toggle
    col_bio1, col_bio2 = st.columns([1, 2])
    with col_bio1:
        use_biobert = st.checkbox(
            "🧠 Use BioBERT NER", 
            value=st.session_state.use_biobert,
            help="Use transformer-based NER for better entity extraction (slower, requires ~2GB)"
        )
        st.session_state.use_biobert = use_biobert
    with col_bio2:
        if use_biobert:
            if TRANSFORMERS_AVAILABLE:
                st.success("✅ BioBERT available")
            else:
                st.warning("⚠️ Install transformers: `pip install transformers`")
    
    if st.button("🔍 Search PubMed", type="primary"):
        import time
        start_time = time.time()
        
        with st.spinner("Fetching papers and extracting entities..."):
            # Use LiteratureMiner with BioBERT option
            effective_biobert = use_biobert and TRANSFORMERS_AVAILABLE
            miner = LiteratureMiner(use_biobert=effective_biobert)
            lit_results = miner.search_and_extract(query, max_results=max_papers)
            papers = lit_results['papers']
            
            # Extract relations
            relations = miner.extract_relations(lit_results)
            
            # Build knowledge graph
            kg = KnowledgeGraph()
            kg.from_entities(lit_results['entities'], papers)
            
            st.session_state.literature_results = papers
            st.session_state.literature_ner = {
                'entities': lit_results['entities'],
                'entity_counts': lit_results['entity_counts'],
                'unique_entities': lit_results['unique_entities'],
                'top_entities': lit_results['top_entities'],
                'relations': relations,
                'knowledge_graph': kg
            }
            
            # Store in database
            db = st.session_state.db
            execution_time = time.time() - start_time
            
            for paper in papers:
                db.insert_paper({
                    'pmid': paper.get('pmid'),
                    'title': paper.get('title'),
                    'authors': paper.get('authors'),
                    'journal': paper.get('journal'),
                    'abstract': paper.get('abstract'),
                    'query_used': query
                })
                
                # Store entities for this paper
                for entity in paper.get('entities', []):
                    db.insert_entity({
                        'name': entity.get('text'),
                        'entity_type': entity.get('type'),
                        'source_pmid': paper.get('pmid'),
                        'confidence': entity.get('confidence', 1.0)
                    })
            
            # Store relations
            for rel in relations:
                db.insert_relation({
                    'source_entity': rel.get('source'),
                    'relation_type': rel.get('relation', 'ASSOCIATED_WITH'),
                    'target_entity': rel.get('target'),
                    'confidence': rel.get('confidence', 1.0)
                })
            
            # Log query
            db.log_query(query, 'pubmed', len(papers), execution_time)
            
            st.toast(f"💾 Saved {len(papers)} papers to database")
    
    # Display results
    if st.session_state.literature_results:
        papers = st.session_state.literature_results
        ner_data = st.session_state.literature_ner
        
        st.success(f"✅ Found {len(papers)} papers")
        
        # NER Statistics Panel
        if ner_data:
            st.markdown("### 🏷️ Extracted Entities")
            
            entity_counts = ner_data.get('entity_counts', {})
            top_entities = ner_data.get('top_entities', [])
            relations = ner_data.get('relations', [])
            kg = ner_data.get('knowledge_graph')
            
            # Entity type counts
            if entity_counts:
                cols = st.columns(len(entity_counts))
                color_map = {'GENE': '🔴', 'DRUG': '🔵', 'DISEASE': '🟢', 'MUTATION': '🟡', 'PATHWAY': '🟣'}
                for i, (etype, count) in enumerate(entity_counts.items()):
                    with cols[i % len(cols)]:
                        icon = color_map.get(etype, '⚪')
                        st.metric(f"{icon} {etype}", count)
            
            # Top entities
            if top_entities:
                st.markdown("**Most Mentioned Entities:**")
                entity_text = " • ".join([f"`{e}` ({c})" for e, c in top_entities[:10]])
                st.markdown(entity_text)
            
            # Gene-Disease Relations
            gene_disease = [(r['source'], r['target']) for r in relations 
                           if r.get('source_type') == 'GENE' and r.get('target_type') == 'DISEASE'][:5]
            
            if gene_disease:
                st.markdown("**Gene-Disease Associations:**")
                for gene, disease in gene_disease:
                    st.markdown(f"- **{gene}** → {disease}")
            
            # Knowledge Graph visualization
            if kg and kg.nodes:
                stats = kg.get_statistics()
                st.info(f"🕸️ Knowledge Graph: {stats['total_nodes']} nodes, {stats['total_edges']} edges")
                
                # Render interactive graph
                with st.expander("🕸️ View Interactive Knowledge Graph", expanded=False):
                    try:
                        import streamlit.components.v1 as components
                        import tempfile
                        import os
                        
                        # Generate PyVis network
                        net = kg.to_pyvis(height="500px", width="100%", physics=True)
                        
                        if net is not None:
                            # Save to temp file
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w') as f:
                                net.save_graph(f.name)
                                temp_path = f.name
                            
                            # Read and display
                            with open(temp_path, 'r', encoding='utf-8') as f:
                                html_content = f.read()
                            
                            components.html(html_content, height=520, scrolling=True)
                            
                            # Cleanup
                            os.unlink(temp_path)
                        else:
                            st.warning("PyVis not available. Install with: `pip install pyvis`")
                    except Exception as e:
                        st.error(f"Could not render graph: {e}")
            
            st.markdown("---")
        
        # Papers with highlighted entities
        st.markdown("### 📄 Papers")
        for i, paper in enumerate(papers, 1):
            with st.expander(f"📄 {i}. {paper['title'][:80]}...", expanded=(i==1)):
                st.markdown(f"**Authors:** {paper['authors']}")
                st.markdown(f"**Journal:** {paper['journal']}")
                st.markdown(f"**PMID:** {paper['pmid']}")
                
                # Show entities for this paper
                paper_entities = paper.get('entities', [])
                if paper_entities:
                    entity_tags = []
                    color_map = {'GENE': '#FF6B6B', 'DRUG': '#45B7D1', 'DISEASE': '#96CEB4', 'MUTATION': '#FFEAA7'}
                    for e in paper_entities[:12]:
                        color = color_map.get(e['type'], '#95A5A6')
                        entity_tags.append(f'<span style="background: {color}; color: white; padding: 2px 6px; border-radius: 4px; margin: 2px; font-size: 0.8rem;">{e["text"]}</span>')
                    st.markdown(f"**Entities:** {''.join(entity_tags)}", unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Highlighted abstract
                if paper_entities and paper.get('abstract'):
                    st.markdown("**Abstract (with highlighted entities):**")
                    highlighted = highlight_entities(paper['abstract'], paper_entities)
                    st.markdown(highlighted, unsafe_allow_html=True)
                else:
                    st.markdown("**Abstract:**")
                    st.write(paper['abstract'])
                
                if paper['pmid'] != 'N/A':
                    st.markdown(f"[🔗 View on PubMed](https://pubmed.ncbi.nlm.nih.gov/{paper['pmid']}/)")

# ============================================================
# TAB 2: IMAGING ANALYSIS
# ============================================================

with tab2:
    st.header("🔬 Imaging Analysis")
    st.write("Upload a histopathology image for cancer detection with GradCAM explainability.")
    
    uploaded_image = st.file_uploader(
        "Upload histopathology image (PNG, JPG)",
        type=['png', 'jpg', 'jpeg'],
        key="image_uploader"
    )
    
    if uploaded_image:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            img = Image.open(uploaded_image)
            st.image(img, use_container_width=True)
        
        if st.button("🔬 Analyze Image", type="primary"):
            with st.spinner("Running ResNet50 + GradCAM analysis..."):
                uploaded_image.seek(0)
                img_pil = Image.open(uploaded_image)
                pred, conf, heatmap = imaging.predict_image(img_pil)
                
                st.session_state.imaging_results = {
                    'prediction': pred,
                    'confidence': conf,
                    'heatmap': heatmap
                }
                
                # Store prediction in database
                db = st.session_state.db
                db.insert_prediction({
                    'image_path': uploaded_image.name,
                    'prediction_class': pred,
                    'confidence': conf,
                    'model_used': 'ResNet50+GradCAM'
                })
                st.toast("💾 Prediction saved to database")
        
        # Display results
        if st.session_state.imaging_results:
            results = st.session_state.imaging_results
            
            with col2:
                st.subheader("GradCAM Heatmap")
                st.image(results['heatmap'], use_container_width=True)
            
            st.markdown("---")
            
            # Result display
            pred_class = "result-cancer" if results['prediction'] == "Malignant" else "result-normal"
            st.markdown(f"""
            <div class="result-box {pred_class}">
                <h3>Prediction: {results['prediction']}</h3>
                <p>Confidence: {results['confidence']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.success("✅ Imaging analysis complete! Go to Tab 3 for Omics Analysis.")
    else:
        st.info("👆 Upload a histopathology image to begin analysis")
        
        # Show sample images location
        st.markdown("""
        **Sample images available at:**
        - `data/Breakhis-400x/benign/` - Benign tissue samples
        - `data/Breakhis-400x/malignant/` - Malignant tissue samples
        """)

# ============================================================
# TAB 3: OMICS ANALYSIS
# ============================================================

with tab3:
    st.header("🧬 Omics Analysis")
    st.write("Upload gene expression data for molecular classification and biomarker identification.")
    
    uploaded_csv = st.file_uploader(
        "Upload gene expression CSV",
        type=['csv'],
        key="csv_uploader"
    )
    
    if uploaded_csv:
        # Preview data
        uploaded_csv.seek(0)
        df_preview = pd.read_csv(uploaded_csv, nrows=5)
        st.subheader("Data Preview")
        st.dataframe(df_preview, use_container_width=True)
        st.caption(f"Showing 5 of {len(df_preview)} rows, {len(df_preview.columns)} columns")
        
        if st.button("🧬 Analyze Gene Expression", type="primary"):
            with st.spinner("Running ML pipeline: Split → Scale → Select → Classify..."):
                uploaded_csv.seek(0)
                pred, conf, biomarkers = omics.analyze_omics_file(uploaded_csv)
                
                st.session_state.omics_results = {
                    'prediction': pred,
                    'confidence': conf,
                    'biomarkers': biomarkers
                }
                
                # Store biomarkers in database
                db = st.session_state.db
                pred_id = db.insert_prediction({
                    'image_path': uploaded_csv.name,
                    'prediction_class': pred,
                    'confidence': conf,
                    'model_used': 'RandomForest-Omics'
                })
                db.insert_biomarkers(biomarkers, prediction_id=pred_id)
                st.toast(f"💾 Saved {len(biomarkers)} biomarkers to database")
        
        # Display results
        if st.session_state.omics_results:
            results = st.session_state.omics_results
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Classification Result")
                pred_class = "result-cancer" if results['prediction'] == "Cancer" else "result-normal"
                st.markdown(f"""
                <div class="result-box {pred_class}">
                    <h3>Prediction: {results['prediction']}</h3>
                    <p>Confidence: {results['confidence']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.subheader("Top Biomarkers")
                for i, (gene, importance) in enumerate(results['biomarkers'][:5], 1):
                    st.markdown(f"""
                    <div class="biomarker-item">
                        <strong>{i}. {gene}</strong> - Importance: {importance:.4f}
                    </div>
                    """, unsafe_allow_html=True)
            
            st.success("✅ Omics analysis complete! Go to Tab 4 for Integrated Results.")
    else:
        st.info("👆 Upload a gene expression CSV file")
        
        st.markdown("""
        **Sample data available at:**
        - `data/tcga_brca_key_genes.csv` - TCGA breast cancer key genes (24 genes)
        - `data/tcga_brca_top500.csv` - TCGA top 500 variable genes
        
        **Expected format:**
        - Rows = samples
        - Columns = genes + 'label' column
        - Labels: 'Cancer' or 'Normal'
        """)

# ============================================================
# TAB 4: INTEGRATED RESULTS
# ============================================================

with tab4:
    st.header("📊 Integrated Multimodal Results")
    
    # Check if all analyses are done
    has_literature = st.session_state.literature_results is not None
    has_imaging = st.session_state.imaging_results is not None
    has_omics = st.session_state.omics_results is not None
    
    # Status indicators
    col1, col2, col3 = st.columns(3)
    with col1:
        if has_literature:
            st.success("✅ Literature: Complete")
        else:
            st.warning("⏳ Literature: Pending")
    with col2:
        if has_imaging:
            st.success("✅ Imaging: Complete")
        else:
            st.warning("⏳ Imaging: Pending")
    with col3:
        if has_omics:
            st.success("✅ Omics: Complete")
        else:
            st.warning("⏳ Omics: Pending")
    
    st.markdown("---")
    
    if has_imaging and has_omics:
        img_results = st.session_state.imaging_results
        omics_results = st.session_state.omics_results
        papers = st.session_state.literature_results or []
        
        # Get biomarker names
        biomarkers = omics_results['biomarkers']
        
        # Generate integrated summary
        summary = integration.generate_summary(
            img_results['prediction'], img_results['confidence'],
            omics_results['prediction'], omics_results['confidence'],
            biomarkers, papers
        )
        
        # ==================== DECISION PANEL ====================
        st.subheader("🧠 Clinical Decision Summary")
        
        risk_class = "risk-high" if summary['risk_score'] > 0.85 else "risk-moderate" if summary['risk_score'] > 0.65 else "risk-low"
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Risk Level</h4>
                <p class="{risk_class}">{summary['risk_level']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Risk Score</h4>
                <p>{summary['risk_score']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Imaging</h4>
                <p>{img_results['prediction']}<br>{img_results['confidence']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Omics</h4>
                <p>{omics_results['prediction']}<br>{omics_results['confidence']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ==================== DETAILED SUMMARY ====================
        st.subheader("📋 Detailed Analysis")
        st.markdown(summary['summary_text'])
        
        st.markdown("---")
        
        # ==================== SIDE BY SIDE RESULTS ====================
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔬 Imaging Result")
            st.image(img_results['heatmap'], caption="GradCAM Attention Map", use_container_width=True)
        
        with col2:
            st.subheader("🧬 Top Biomarkers")
            for i, (gene, imp) in enumerate(biomarkers[:8], 1):
                pct = min(imp * 100 / 30, 1.0)  # Clamp to [0, 1]
                st.progress(pct, text=f"{i}. **{gene}**: {imp:.4f}")
        
        # ==================== LITERATURE EVIDENCE ====================
        if papers:
            st.markdown("---")
            st.subheader("📚 Supporting Literature")
            for paper in papers[:3]:
                st.markdown(f"""
                <div class="paper-card">
                    <strong>{paper['title']}</strong><br>
                    <small>{paper['authors']} | {paper['journal']} | PMID: {paper['pmid']}</small>
                </div>
                """, unsafe_allow_html=True)
        
        # ==================== DISCLAIMER ====================
        st.markdown("---")
        st.warning("""
        ⚕️ **Clinical Disclaimer**: This AI system is for research and clinical decision support only. 
        It is NOT a substitute for professional medical diagnosis. All findings should be validated 
        by qualified healthcare professionals before any clinical decisions are made.
        """)
        
    else:
        st.info("""
        **Complete the following steps to see integrated results:**
        
        1. **Tab 1 - Literature:** Search for relevant papers (optional but recommended)
        2. **Tab 2 - Imaging:** Upload and analyze a histopathology image
        3. **Tab 3 - Omics:** Upload and analyze gene expression data
        
        Once both Imaging and Omics are complete, the integrated analysis will appear here.
        """)

# ============================================================
# TAB 5: ADVANCED INSIGHTS (NEW FEATURES)
# ============================================================

with tab5:
    st.header("🧪 Advanced Insights")
    st.write("Molecular subtyping, treatment recommendations, clinical trials, and report generation.")
    
    # Check prerequisites
    has_omics = st.session_state.omics_results is not None
    has_imaging = st.session_state.imaging_results is not None
    
    if not has_omics:
        st.warning("⚠️ Complete Omics Analysis (Tab 3) first to unlock advanced insights.")
        st.info("The molecular subtype classifier and treatment recommendations require biomarker data from omics analysis.")
    else:
        omics_results = st.session_state.omics_results
        biomarkers = omics_results['biomarkers']
        biomarker_dict = {gene: imp for gene, imp in biomarkers}
        
        # Sub-tabs for advanced features
        adv_tab1, adv_tab2, adv_tab3, adv_tab4 = st.tabs([
            "🎯 Molecular Subtype",
            "💊 Treatment Recommendations",
            "🔬 Clinical Trials",
            "📄 Generate Report"
        ])
        
        # ==================== MOLECULAR SUBTYPE ====================
        with adv_tab1:
            st.subheader("🎯 PAM50-Style Molecular Subtype Classification")
            st.write("Classify breast cancer into molecular subtypes based on gene expression patterns.")
            
            # Optional clinical input
            with st.expander("➕ Add Clinical IHC Data (Optional)", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    er_status = st.selectbox("ER Status", ["Unknown", "Positive", "Negative"])
                with col2:
                    pr_status = st.selectbox("PR Status", ["Unknown", "Positive", "Negative"])
                with col3:
                    her2_status = st.selectbox("HER2 Status", ["Unknown", "Positive", "Negative"])
                    
                er_val = er_status.lower() if er_status != "Unknown" else None
                pr_val = pr_status.lower() if pr_status != "Unknown" else None
                her2_val = her2_status.lower() if her2_status != "Unknown" else None
            
            if st.button("🧬 Classify Molecular Subtype", type="primary", key="classify_btn"):
                with st.spinner("Analyzing gene expression patterns..."):
                    subtype, probabilities, analysis = classify_subtype(
                        biomarker_dict,
                        er_status=er_val if 'er_val' in dir() else None,
                        pr_status=pr_val if 'pr_val' in dir() else None,
                        her2_status=her2_val if 'her2_val' in dir() else None
                    )
                    st.session_state.subtype_result = {
                        'subtype': subtype,
                        'probabilities': probabilities,
                        'analysis': analysis
                    }
            
            # Display results
            if st.session_state.subtype_result:
                result = st.session_state.subtype_result
                subtype = result['subtype']
                probs = result['probabilities']
                
                # Main result
                color = get_subtype_color(subtype)
                st.markdown(f"""
                <div style="text-align: center; padding: 30px; background: {color}20; 
                     border: 3px solid {color}; border-radius: 15px; margin: 20px 0;">
                    <h2 style="color: {color}; margin: 0;">🎯 {subtype}</h2>
                    <p style="font-size: 18px; margin: 10px 0;">Confidence: {probs.get(subtype, 0):.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Subtype info
                info = SUBTYPE_INFO.get(subtype, {})
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Characteristics:**")
                    for char in info.get('characteristics', [])[:4]:
                        st.markdown(f"• {char}")
                        
                with col2:
                    st.markdown("**Prognosis:**")
                    st.info(info.get('prognosis', 'N/A'))
                    st.markdown(f"**Prevalence:** {info.get('prevalence', 'N/A')}")
                
                # Probability distribution
                st.markdown("---")
                st.markdown("**Probability Distribution:**")
                for st_name, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                    st.progress(prob, text=f"{st_name}: {prob:.1%}")
        
        # ==================== TREATMENT RECOMMENDATIONS ====================
        with adv_tab2:
            st.subheader("💊 AI-Powered Treatment Recommendations")
            st.write("Evidence-based therapy suggestions based on your molecular profile.")
            
            if not st.session_state.subtype_result:
                st.info("👆 First classify the molecular subtype in the previous tab.")
            else:
                subtype = st.session_state.subtype_result['subtype']
                
                # Clinical factors
                with st.expander("➕ Add Clinical Factors (Optional)", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        stage = st.selectbox("Disease Stage", ["Unknown", "I", "II", "III", "IV"])
                        menopause = st.selectbox("Menopausal Status", ["Unknown", "Pre", "Post"])
                    with col2:
                        brca = st.selectbox("BRCA Status", ["Unknown", "Mutated", "Wild-type"])
                        pdl1 = st.selectbox("PD-L1 Status", ["Unknown", "Positive", "Negative"])
                        
                stage_val = stage if stage != "Unknown" else None
                menopause_val = menopause.lower() if menopause != "Unknown" else None
                brca_val = 'mutated' if brca == "Mutated" else ('wild-type' if brca == "Wild-type" else None)
                pdl1_val = pdl1.lower() if pdl1 != "Unknown" else None
                
                if st.button("💊 Generate Treatment Recommendations", type="primary", key="treatment_btn"):
                    with st.spinner("Analyzing treatment options..."):
                        recs = get_treatment_recommendations(
                            subtype, biomarker_dict,
                            stage=stage_val,
                            menopausal_status=menopause_val,
                            brca_status=brca_val,
                            pd_l1_status=pdl1_val
                        )
                        st.session_state.treatment_recs = recs
                
                # Display recommendations
                if st.session_state.treatment_recs:
                    recs = st.session_state.treatment_recs
                    
                    # Rationale
                    if recs.get('rationale'):
                        st.markdown("**Clinical Rationale:**")
                        for r in recs['rationale']:
                            st.info(r)
                    
                    # Primary recommendations
                    st.markdown("---")
                    st.markdown("### 🥇 Primary Recommendations")
                    for tx in recs.get('primary', [])[:4]:
                        treatment = tx.get('treatment', 'Unknown')
                        details = tx.get('details')
                        
                        with st.container():
                            st.markdown(f"""
                            <div style="padding: 15px; background: #e3f2fd; border-left: 4px solid #2196F3; 
                                 border-radius: 5px; margin: 10px 0;">
                                <strong style="font-size: 16px;">💊 {treatment}</strong><br>
                                <em>{tx.get('rationale', '')}</em>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if hasattr(details, 'mechanism'):
                                with st.expander(f"ℹ️ Details: {treatment}"):
                                    st.markdown(f"**Mechanism:** {details.mechanism}")
                                    st.markdown(f"**Evidence Level:** {details.evidence_level}")
                                    st.markdown(f"**Side Effects:** {', '.join(details.side_effects[:3])}")
                    
                    # Additional tests
                    if recs.get('biomarker_tests_recommended'):
                        st.markdown("---")
                        st.markdown("### 🔬 Recommended Additional Tests")
                        for test in recs['biomarker_tests_recommended']:
                            st.markdown(f"• {test}")
                    
                    # Warnings
                    if recs.get('warnings'):
                        st.markdown("---")
                        for warning in recs['warnings']:
                            st.warning(warning)
        
        # ==================== CLINICAL TRIALS ====================
        with adv_tab3:
            st.subheader("🔬 Clinical Trial Matcher")
            st.write("Find clinical trials matching your molecular profile.")
            
            if not st.session_state.subtype_result:
                st.info("👆 First classify the molecular subtype to find matching trials.")
            else:
                subtype = st.session_state.subtype_result['subtype']
                
                # Options
                col1, col2 = st.columns(2)
                with col1:
                    include_api = st.checkbox("🌐 Search live ClinicalTrials.gov", value=False, 
                                             help="Search real-time API (may take longer)")
                with col2:
                    her2_for_trials = st.selectbox("HER2 Status for matching", 
                                                   ["Unknown", "Positive", "Negative"], key="her2_trials")
                
                if st.button("🔍 Find Matching Trials", type="primary", key="trials_btn"):
                    with st.spinner("Searching for matching clinical trials..."):
                        trials = find_matching_trials(
                            subtype, biomarker_dict,
                            her2_status=her2_for_trials.lower() if her2_for_trials != "Unknown" else None,
                            include_api_search=include_api
                        )
                        st.session_state.matched_trials = trials
                
                # Display trials
                if st.session_state.matched_trials:
                    trials = st.session_state.matched_trials
                    
                    st.success(f"Found {len(trials)} matching clinical trials!")
                    
                    for i, trial in enumerate(trials[:8], 1):
                        status_color = "#4CAF50" if "Recruiting" in trial.status else "#FF9800"
                        
                        st.markdown(f"""
                        <div style="padding: 15px; background: #fafafa; border: 1px solid #e0e0e0;
                             border-radius: 8px; margin: 10px 0;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="background: {status_color}; color: white; padding: 3px 10px; 
                                      border-radius: 12px; font-size: 12px;">{trial.status}</span>
                                <span style="color: #666;">Match: {trial.match_score:.0%}</span>
                            </div>
                            <h4 style="margin: 10px 0 5px 0;">{trial.title}</h4>
                            <p style="color: #666; margin: 5px 0;">
                                <strong>NCT ID:</strong> {trial.nct_id} | 
                                <strong>Phase:</strong> {trial.phase}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander(f"📋 Details: {trial.nct_id}"):
                            st.markdown(f"**Summary:** {trial.brief_summary[:300]}...")
                            st.markdown(f"**Interventions:** {', '.join(trial.interventions[:3])}")
                            st.markdown(f"**Sponsor:** {trial.sponsor}")
                            if trial.match_reasons:
                                st.markdown("**Why this matches:**")
                                for reason in trial.match_reasons:
                                    st.markdown(f"• {reason}")
                            st.markdown(f"🔗 [View on ClinicalTrials.gov]({trial.url})")
        
        # ==================== REPORT GENERATOR ====================
        with adv_tab4:
            st.subheader("📄 Generate Diagnostic Report")
            st.write("Create a comprehensive PDF/HTML report with all analysis results.")
            
            # Check what's available
            status_items = []
            if st.session_state.literature_results:
                status_items.append("✅ Literature data")
            if st.session_state.imaging_results:
                status_items.append("✅ Imaging results")
            if st.session_state.omics_results:
                status_items.append("✅ Omics results")
            if st.session_state.subtype_result:
                status_items.append("✅ Molecular subtype")
            if st.session_state.treatment_recs:
                status_items.append("✅ Treatment recommendations")
            if st.session_state.matched_trials:
                status_items.append("✅ Clinical trials")
                
            st.markdown("**Data available for report:**")
            for item in status_items:
                st.markdown(item)
            
            if len(status_items) < 3:
                st.warning("Complete more analyses to generate a comprehensive report.")
            
            st.markdown("---")
            
            report_format = st.radio("Report Format", ["HTML (viewable in browser)", "PDF (if available)"])
            
            if st.button("📄 Generate Report", type="primary", key="report_btn"):
                with st.spinner("Generating comprehensive report..."):
                    # Gather all data
                    img_res = st.session_state.imaging_results
                    omics_res = st.session_state.omics_results
                    
                    report = create_report(
                        imaging_result=img_res['prediction'] if img_res else None,
                        imaging_confidence=img_res['confidence'] if img_res else None,
                        omics_result=omics_res['prediction'] if omics_res else None,
                        omics_confidence=omics_res['confidence'] if omics_res else None,
                        biomarkers=biomarkers if omics_res else [],
                        molecular_subtype=st.session_state.subtype_result['subtype'] if st.session_state.subtype_result else None,
                        subtype_probabilities=st.session_state.subtype_result['probabilities'] if st.session_state.subtype_result else {},
                        treatment_recommendations=st.session_state.treatment_recs,
                        clinical_trials=st.session_state.matched_trials or [],
                        literature_papers=st.session_state.literature_results or [],
                        risk_level="HIGH" if (img_res and img_res['prediction'] == 'Malignant') else "MODERATE",
                        summary_text="Comprehensive AI-powered diagnostic analysis combining histopathology imaging, gene expression profiling, and literature evidence."
                    )
                    
                    # Generate report
                    if "PDF" in report_format and is_pdf_available():
                        report_content = export_report(report, 'pdf')
                        st.download_button(
                            "⬇️ Download PDF Report",
                            report_content,
                            file_name=f"breast_cancer_report_{report.report_id}.pdf",
                            mime="application/pdf"
                        )
                    else:
                        report_content = export_report(report, 'html')
                        
                        # Show preview
                        st.markdown("### Report Preview")
                        st.components.v1.html(report_content, height=600, scrolling=True)
                        
                        # Download button
                        st.download_button(
                            "⬇️ Download HTML Report",
                            report_content,
                            file_name=f"breast_cancer_report_{report.report_id}.html",
                            mime="text/html"
                        )
                    
                    st.success("✅ Report generated successfully!")

# ============================================================
# TAB 6: DATABASE HISTORY
# ============================================================

with tab6:
    st.header("🗄️ Database History")
    st.write("Browse stored papers, entities, and analysis history.")
    
    db = st.session_state.db
    
    # Database stats
    stats = db.get_database_stats()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📄 Papers", stats.get('papers', 0))
    with col2:
        st.metric("🏷️ Entities", stats.get('entities', 0))
    with col3:
        st.metric("🔗 Relations", stats.get('relations', 0))
    with col4:
        st.metric("🔬 Predictions", stats.get('predictions', 0))
    with col5:
        st.metric("🧬 Biomarkers", stats.get('biomarkers', 0))
    
    st.markdown("---")
    
    # Sub-tabs for different data types
    hist_tab1, hist_tab2, hist_tab3, hist_tab4 = st.tabs([
        "📄 Papers", "🏷️ Entities", "🔗 Relations", "📊 Query History"
    ])
    
    with hist_tab1:
        st.subheader("Stored Papers")
        papers = db.get_all_papers(limit=50)
        if papers:
            for paper in papers:
                with st.expander(f"📄 {paper.get('title', 'Untitled')[:60]}..."):
                    st.markdown(f"**PMID:** {paper.get('pmid', 'N/A')}")
                    st.markdown(f"**Authors:** {paper.get('authors', 'N/A')}")
                    st.markdown(f"**Journal:** {paper.get('journal', 'N/A')}")
                    st.markdown(f"**Query:** {paper.get('query_used', 'N/A')}")
                    st.markdown(f"**Stored:** {paper.get('created_at', 'N/A')}")
                    if paper.get('abstract'):
                        st.text_area("Abstract", paper['abstract'], height=100, disabled=True)
        else:
            st.info("No papers stored yet. Search PubMed in Tab 1 to populate.")
    
    with hist_tab2:
        st.subheader("Top Entities")
        
        # Filter by type
        entity_stats = db.get_entity_stats()
        entity_types = list(entity_stats.keys()) if entity_stats else []
        
        if entity_types:
            selected_type = st.selectbox("Filter by type:", ["All"] + entity_types)
            
            if selected_type == "All":
                top_entities = db.get_top_entities(limit=30)
            else:
                top_entities = db.get_top_entities(entity_type=selected_type, limit=30)
            
            if top_entities:
                # Create dataframe for display
                df_entities = pd.DataFrame(top_entities)
                df_entities.columns = ['Name', 'Type', 'Count', 'Avg Confidence']
                st.dataframe(df_entities, use_container_width=True)
            else:
                st.info("No entities found.")
        else:
            st.info("No entities stored yet. Search PubMed to extract entities.")
    
    with hist_tab3:
        st.subheader("Entity Relations")
        relations = db.get_all_relations(limit=50)
        
        if relations:
            for rel in relations[:20]:
                st.markdown(
                    f"**{rel['source_entity']}** → *{rel['relation_type']}* → **{rel['target_entity']}** "
                    f"(conf: {rel.get('confidence', 1.0):.2f})"
                )
        else:
            st.info("No relations stored yet.")
    
    with hist_tab4:
        st.subheader("Query History")
        history = db.get_query_history(limit=20)
        
        if history:
            for q in history:
                st.markdown(
                    f"🔍 **{q['query_text'][:50]}...** | "
                    f"Results: {q.get('results_count', 0)} | "
                    f"Time: {q.get('execution_time', 0):.2f}s | "
                    f"{q.get('created_at', '')}"
                )
        else:
            st.info("No queries logged yet.")
    
    st.markdown("---")
    
    # Export and management
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📥 Export Entities CSV"):
            try:
                export_path = "database/exported_entities.csv"
                db.export_entities_csv(export_path)
                st.success(f"Exported to {export_path}")
            except Exception as e:
                st.error(f"Export failed: {e}")
    
    with col2:
        if st.button("📥 Export Relations CSV"):
            try:
                export_path = "database/exported_relations.csv"
                db.export_relations_csv(export_path)
                st.success(f"Exported to {export_path}")
            except Exception as e:
                st.error(f"Export failed: {e}")
    
    with col3:
        if st.button("🗑️ Clear Database", type="secondary"):
            if st.session_state.get('confirm_clear', False):
                db.clear_all()
                st.success("Database cleared!")
                st.session_state.confirm_clear = False
                st.rerun()
            else:
                st.session_state.confirm_clear = True
                st.warning("Click again to confirm clearing all data")

# ============================================================
# SIDEBAR - Quick Status
# ============================================================

with st.sidebar:
    st.header("📋 Analysis Status")
    
    st.markdown("---")
    
    if st.session_state.literature_results:
        ner_data = st.session_state.literature_ner
        entities_count = len(ner_data.get('entities', [])) if ner_data else 0
        st.success(f"📚 Literature: {len(st.session_state.literature_results)} papers")
        if entities_count > 0:
            st.caption(f"🏷️ {entities_count} entities extracted")
    else:
        st.info("📚 Literature: Not started")
    
    if st.session_state.imaging_results:
        r = st.session_state.imaging_results
        st.success(f"🔬 Imaging: {r['prediction']} ({r['confidence']:.0%})")
    else:
        st.info("🔬 Imaging: Not started")
    
    if st.session_state.omics_results:
        r = st.session_state.omics_results
        st.success(f"🧬 Omics: {r['prediction']} ({r['confidence']:.0%})")
    else:
        st.info("🧬 Omics: Not started")
    
    # Advanced insights status
    st.markdown("---")
    st.markdown("**Advanced Insights:**")
    
    if st.session_state.subtype_result:
        st.success(f"🎯 Subtype: {st.session_state.subtype_result['subtype']}")
    else:
        st.info("🎯 Subtype: Pending")
        
    if st.session_state.treatment_recs:
        st.success(f"💊 Treatments: Ready")
    else:
        st.info("💊 Treatments: Pending")
        
    if st.session_state.matched_trials:
        st.success(f"🔬 Trials: {len(st.session_state.matched_trials)} matched")
    else:
        st.info("🔬 Trials: Pending")
    
    st.markdown("---")
    
    if st.button("🔄 Reset All", type="secondary"):
        st.session_state.literature_results = None
        st.session_state.literature_ner = None
        st.session_state.imaging_results = None
        st.session_state.omics_results = None
        st.session_state.subtype_result = None
        st.session_state.treatment_recs = None
        st.session_state.matched_trials = None
        st.rerun()
    
    st.markdown("---")
    st.caption("Breast Cancer AI v4.0")
    st.caption("Advanced Insights Edition")
    st.caption("For Research Use Only")
