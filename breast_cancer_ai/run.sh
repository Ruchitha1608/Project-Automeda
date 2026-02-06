#!/bin/bash
# Quick start script for Breast Cancer AI application

echo "🧬 Starting Explainable Multimodal AI for Breast Cancer Detection..."
echo ""
echo "📍 Project: breast_cancer_ai/"
echo "🌐 Opening in browser at: http://localhost:8501"
echo ""
echo "💡 To use:"
echo "   1. Upload data/sample_breast.jpg as the histopathology image"
echo "   2. Upload data/sample_omics.csv as the gene expression data"
echo "   3. Click 'RUN ANALYSIS'"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run app.py --server.port 8501
