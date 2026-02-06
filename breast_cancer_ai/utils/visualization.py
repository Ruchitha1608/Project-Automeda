"""
Visualization Utilities
Comprehensive plotting functions for the diagnostic system
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union
import io
import base64


# ============================================================
# HEATMAP VISUALIZATION
# ============================================================

def create_heatmap_overlay(
    original_image: Union[np.ndarray, Image.Image],
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = 'jet'
) -> Image.Image:
    """
    Create heatmap overlay on original image
    
    Args:
        original_image: Original image (PIL or numpy)
        heatmap: Attention/activation map (2D array, values 0-1)
        alpha: Overlay transparency (0-1)
        colormap: Matplotlib colormap name
    
    Returns:
        PIL Image with heatmap overlay
    """
    # Convert to numpy if PIL
    if isinstance(original_image, Image.Image):
        original = np.array(original_image)
    else:
        original = original_image.copy()
    
    # Ensure original is RGB
    if len(original.shape) == 2:
        original = np.stack([original] * 3, axis=-1)
    elif original.shape[-1] == 4:
        original = original[:, :, :3]
    
    # Resize heatmap to match original
    from PIL import Image as PILImage
    heatmap_img = PILImage.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_resized = heatmap_img.resize((original.shape[1], original.shape[0]), PILImage.Resampling.BILINEAR)
    heatmap = np.array(heatmap_resized) / 255.0
    
    # Apply colormap
    cmap = plt.cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # Remove alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Blend images
    original_float = original.astype(np.float32)
    heatmap_float = heatmap_colored.astype(np.float32)
    
    blended = (1 - alpha) * original_float + alpha * heatmap_float
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    return PILImage.fromarray(blended)


def create_gradcam_visualization(
    image: Image.Image,
    heatmap: np.ndarray,
    prediction: str,
    confidence: float
) -> Image.Image:
    """
    Create GradCAM visualization with annotations
    
    Args:
        image: Original image
        heatmap: GradCAM heatmap
        prediction: Prediction label
        confidence: Confidence score
    
    Returns:
        Annotated visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    # Heatmap
    im = axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('GradCAM Heatmap', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    overlay = create_heatmap_overlay(image, heatmap)
    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay\n{prediction} ({confidence:.1%})', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Convert to PIL
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)


# ============================================================
# CONFIDENCE VISUALIZATION
# ============================================================

def create_confidence_gauge(
    confidence: float,
    label: str = "Confidence",
    figsize: Tuple[int, int] = (6, 3)
) -> Image.Image:
    """
    Create a gauge-style confidence visualization
    
    Args:
        confidence: Confidence value (0-1)
        label: Label text
        figsize: Figure size
    
    Returns:
        PIL Image of gauge
    """
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})
    
    # Create gauge
    theta = np.linspace(0, np.pi, 100)
    
    # Background arc
    ax.fill_between(theta, 0.6, 1.0, color='#E8ECF0', alpha=0.8)
    
    # Colored sections
    colors = ['#E74C3C', '#F39C12', '#F1C40F', '#2ECC71']
    boundaries = [0, 0.25, 0.5, 0.75, 1.0]
    
    for i in range(len(colors)):
        start = np.pi * (1 - boundaries[i+1])
        end = np.pi * (1 - boundaries[i])
        section_theta = np.linspace(start, end, 25)
        ax.fill_between(section_theta, 0.6, 1.0, color=colors[i], alpha=0.6)
    
    # Needle
    needle_angle = np.pi * (1 - confidence)
    ax.annotate('', xy=(needle_angle, 0.9), xytext=(needle_angle, 0.2),
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2))
    
    # Center dot
    ax.scatter([needle_angle], [0.2], s=100, c='#2C3E50', zorder=5)
    
    # Configure
    ax.set_ylim(0, 1.2)
    ax.set_xlim(0, np.pi)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)
    
    # Labels
    plt.title(f"{label}\n{confidence:.1%}", fontsize=14, fontweight='bold', y=0.85)
    
    # Convert to PIL
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', transparent=True)
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)


def create_confidence_bars(
    confidences: Dict[str, float],
    title: str = "Model Confidence",
    figsize: Tuple[int, int] = (8, 4)
) -> Image.Image:
    """
    Create horizontal bar chart for multiple confidences
    
    Args:
        confidences: Dict of {label: confidence}
        title: Chart title
        figsize: Figure size
    
    Returns:
        PIL Image
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    labels = list(confidences.keys())
    values = list(confidences.values())
    
    # Color based on value
    colors = []
    for v in values:
        if v >= 0.8:
            colors.append('#27AE60')
        elif v >= 0.6:
            colors.append('#F39C12')
        else:
            colors.append('#E74C3C')
    
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, edgecolor='white', height=0.6)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.1%}', va='center', fontsize=11, fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel('Confidence', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)


# ============================================================
# BIOMARKER VISUALIZATION
# ============================================================

def create_biomarker_chart(
    biomarkers: List[Tuple[str, float]],
    title: str = "Top Biomarkers",
    top_n: int = 10,
    figsize: Tuple[int, int] = (10, 6)
) -> Image.Image:
    """
    Create bar chart for biomarker importance
    
    Args:
        biomarkers: List of (gene_name, importance) tuples
        title: Chart title
        top_n: Number of top biomarkers to show
        figsize: Figure size
    
    Returns:
        PIL Image
    """
    # Sort and limit
    biomarkers = sorted(biomarkers, key=lambda x: x[1], reverse=True)[:top_n]
    
    genes = [b[0] for b in biomarkers]
    importance = [b[1] for b in biomarkers]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create gradient colors
    cmap = plt.cm.get_cmap('RdYlGn_r')
    colors = [cmap(i / len(genes)) for i in range(len(genes))]
    
    y_pos = np.arange(len(genes))
    bars = ax.barh(y_pos, importance, color=colors, edgecolor='white', height=0.7)
    
    # Add value labels
    for bar, val in zip(bars, importance):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=10)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(genes, fontsize=11, fontfamily='monospace')
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)


def create_expression_heatmap(
    expression_matrix: np.ndarray,
    gene_names: List[str],
    sample_names: List[str] = None,
    title: str = "Gene Expression Heatmap",
    figsize: Tuple[int, int] = (12, 8)
) -> Image.Image:
    """
    Create heatmap of gene expression values
    
    Args:
        expression_matrix: 2D array (samples x genes)
        gene_names: List of gene names
        sample_names: List of sample names (optional)
        title: Chart title
        figsize: Figure size
    
    Returns:
        PIL Image
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Transpose for better visualization (genes as rows)
    data = expression_matrix.T if expression_matrix.shape[1] == len(gene_names) else expression_matrix
    
    # Create heatmap
    im = ax.imshow(data, aspect='auto', cmap='RdBu_r')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Expression Level', fontsize=10)
    
    # Labels
    ax.set_yticks(np.arange(len(gene_names)))
    ax.set_yticklabels(gene_names, fontsize=9, fontfamily='monospace')
    
    if sample_names:
        ax.set_xticks(np.arange(len(sample_names)))
        ax.set_xticklabels(sample_names, rotation=45, ha='right', fontsize=9)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Samples', fontsize=11)
    ax.set_ylabel('Genes', fontsize=11)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)


# ============================================================
# ENTITY / NER VISUALIZATION
# ============================================================

def create_entity_distribution(
    entity_counts: Dict[str, int],
    title: str = "Entity Distribution",
    figsize: Tuple[int, int] = (8, 6)
) -> Image.Image:
    """
    Create pie chart of entity type distribution
    
    Args:
        entity_counts: Dict of {entity_type: count}
        title: Chart title
        figsize: Figure size
    
    Returns:
        PIL Image
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    labels = list(entity_counts.keys())
    sizes = list(entity_counts.values())
    
    # Custom colors
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#95A5A6']
    colors = colors[:len(labels)]
    
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', pctdistance=0.75,
        wedgeprops=dict(width=0.5, edgecolor='white'),
        textprops=dict(fontsize=10)
    )
    
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)


def create_entity_wordcloud(
    entities: List[Tuple[str, int]],
    figsize: Tuple[int, int] = (10, 6)
) -> Image.Image:
    """
    Create word cloud from entities (sized by frequency)
    
    Args:
        entities: List of (entity_name, count) tuples
        figsize: Figure size
    
    Returns:
        PIL Image
    """
    try:
        from wordcloud import WordCloud
        
        # Create frequency dict
        freq = {e[0]: e[1] for e in entities}
        
        wc = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate_from_frequencies(freq)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Entity Word Cloud', fontsize=14, fontweight='bold')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return Image.open(buf)
        
    except ImportError:
        # Fallback to bar chart if wordcloud not installed
        return create_biomarker_chart(entities, title='Entity Frequency', top_n=20)


# ============================================================
# COMPARISON VISUALIZATION
# ============================================================

def create_multimodal_comparison(
    imaging_result: Dict,
    omics_result: Dict,
    figsize: Tuple[int, int] = (12, 5)
) -> Image.Image:
    """
    Create side-by-side comparison of imaging and omics results
    
    Args:
        imaging_result: {'prediction': str, 'confidence': float}
        omics_result: {'prediction': str, 'confidence': float}
        figsize: Figure size
    
    Returns:
        PIL Image
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    results = [
        ('🖼️ Imaging', imaging_result),
        ('🧬 Omics', omics_result)
    ]
    
    for ax, (title, result) in zip(axes, results):
        pred = result['prediction']
        conf = result['confidence']
        
        # Background color based on prediction
        if 'malignant' in pred.lower() or 'cancer' in pred.lower():
            bg_color = '#FADBD8'
            text_color = '#E74C3C'
        else:
            bg_color = '#D5F4E6'
            text_color = '#27AE60'
        
        ax.set_facecolor(bg_color)
        
        # Main text
        ax.text(0.5, 0.7, pred, fontsize=24, fontweight='bold',
                ha='center', va='center', color=text_color,
                transform=ax.transAxes)
        
        ax.text(0.5, 0.4, f'{conf:.1%}', fontsize=32, fontweight='bold',
                ha='center', va='center', color='#2C3E50',
                transform=ax.transAxes)
        
        ax.text(0.5, 0.2, 'Confidence', fontsize=12,
                ha='center', va='center', color='#7F8C8D',
                transform=ax.transAxes)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def pil_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def base64_to_pil(b64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    img_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(img_data))


if __name__ == "__main__":
    # Test visualizations
    print("Testing Visualization Utilities...")
    
    # Test confidence bars
    confidences = {
        'Imaging Model': 0.91,
        'Omics Model': 0.87,
        'Combined': 0.89
    }
    img = create_confidence_bars(confidences)
    print(f"Confidence bars: {img.size}")
    
    # Test biomarker chart
    biomarkers = [
        ('BRCA1', 0.15),
        ('TP53', 0.12),
        ('HER2', 0.10),
        ('ESR1', 0.08),
        ('PIK3CA', 0.07),
    ]
    img = create_biomarker_chart(biomarkers)
    print(f"Biomarker chart: {img.size}")
    
    # Test entity distribution
    entities = {'GENE': 45, 'DRUG': 23, 'DISEASE': 18, 'PROTEIN': 12}
    img = create_entity_distribution(entities)
    print(f"Entity distribution: {img.size}")
    
    # Test multimodal comparison
    img = create_multimodal_comparison(
        {'prediction': 'Malignant', 'confidence': 0.91},
        {'prediction': 'Cancer', 'confidence': 0.87}
    )
    print(f"Multimodal comparison: {img.size}")
    
    print("\n✅ Visualization utilities working correctly!")
