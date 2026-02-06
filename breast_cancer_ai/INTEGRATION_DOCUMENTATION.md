# Integration Module Documentation

## Overview

The **Integration Module** (`modules/integration.py`) is the core decision-making component that combines outputs from all three analysis modules (Literature, Imaging, Omics) to generate a unified clinical risk assessment.

---

## Architecture Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MULTIMODAL INPUT FUSION                          │
└─────────────────────────────────────────────────────────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   LITERATURE  │   │    IMAGING    │   │     OMICS     │
│    MODULE     │   │    MODULE     │   │    MODULE     │
├───────────────┤   ├───────────────┤   ├───────────────┤
│ • PubMed      │   │ • ResNet50    │   │ • RandomForest│
│ • NER/BioBERT │   │ • GradCAM     │   │ • Feature Sel │
│ • KnowledgeGrp│   │ • Heatmap     │   │ • Biomarkers  │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                    │                    │
        │    ┌───────────────┴───────────────┐    │
        │    │        INTEGRATION            │    │
        └───►│          MODULE               │◄───┘
             │   generate_summary()          │
             └───────────────┬───────────────┘
                             │
                             ▼
             ┌───────────────────────────────┐
             │     UNIFIED RISK ASSESSMENT   │
             │  • Risk Score (0-1)           │
             │  • Risk Level (LOW/MOD/HIGH)  │
             │  • Clinical Recommendation    │
             └───────────────────────────────┘
```

---

## Risk Calculation Algorithm

### Input Parameters

| Parameter | Source | Type | Description |
|-----------|--------|------|-------------|
| `img_pred` | Imaging | string | "Benign" or "Malignant" |
| `img_conf` | Imaging | float | Confidence 0.0-1.0 |
| `omics_pred` | Omics | string | "Normal" or "Cancer" |
| `omics_conf` | Omics | float | Confidence 0.0-1.0 |
| `biomarkers` | Omics | list | Top genes by importance |
| `literature` | Literature | list | PubMed papers |

### Decision Matrix

```
┌─────────────────────┬─────────────────────┬─────────────────────────────────┐
│  IMAGING RESULT     │  OMICS RESULT       │  RISK CALCULATION               │
├─────────────────────┼─────────────────────┼─────────────────────────────────┤
│  Malignant          │  Cancer             │  risk = (img_conf + omics_conf) │
│                     │                     │         ─────────────────────── │
│                     │                     │                  2              │
│                     │                     │  → HIGH / MODERATE-HIGH RISK    │
├─────────────────────┼─────────────────────┼─────────────────────────────────┤
│  Malignant          │  Normal             │  risk = img_conf × 0.7          │
│                     │                     │  → MODERATE RISK                │
├─────────────────────┼─────────────────────┼─────────────────────────────────┤
│  Benign             │  Cancer             │  risk = omics_conf × 0.7        │
│                     │                     │  → MODERATE RISK                │
├─────────────────────┼─────────────────────┼─────────────────────────────────┤
│  Benign             │  Normal             │  risk = 1 - avg(confidences)    │
│                     │                     │  → LOW RISK                     │
└─────────────────────┴─────────────────────┴─────────────────────────────────┘
```

### Risk Level Thresholds

```python
if risk_score > 0.85:
    risk_level = "HIGH RISK"
elif risk_score > 0.65:
    risk_level = "MODERATE RISK"  # or "MODERATE-HIGH RISK"
else:
    risk_level = "LOW RISK"
```

---

## Algorithm Pseudocode

```python
def calculate_risk(img_pred, img_conf, omics_pred, omics_conf):
    
    # Step 1: Map predictions to binary cancer indicators
    img_cancer = (img_pred == "Malignant")
    omics_cancer = (omics_pred == "Cancer")
    
    # Step 2: Calculate risk based on agreement
    if img_cancer AND omics_cancer:
        # CONCORDANT POSITIVE: Both indicate cancer
        risk_score = average(img_conf, omics_conf)
        agreement = "Strong multimodal agreement"
        
    elif img_cancer XOR omics_cancer:
        # DISCORDANT: One positive, one negative
        # Weight the positive result at 70%
        risk_score = positive_confidence × 0.7
        agreement = "Mixed signals"
        
    else:
        # CONCORDANT NEGATIVE: Both benign/normal
        # Invert to get "benign confidence"
        risk_score = 1 - average(img_conf, omics_conf)
        agreement = "Multimodal agreement on benign"
    
    # Step 3: Assign risk level
    return risk_score, determine_level(risk_score)
```

---

## Example Calculations

### Example 1: High Risk (Both Positive)
```
Imaging:  Malignant (92% confidence)
Omics:    Cancer    (88% confidence)

Risk Score = (0.92 + 0.88) / 2 = 0.90
Risk Level = HIGH RISK (> 0.85)
Agreement  = Strong multimodal agreement
```

### Example 2: Moderate Risk (Discordant)
```
Imaging:  Malignant (85% confidence)
Omics:    Normal    (78% confidence)

Risk Score = 0.85 × 0.7 = 0.595
Risk Level = MODERATE RISK
Agreement  = Mixed signals - further testing recommended
```

### Example 3: Low Risk (Both Negative)
```
Imaging:  Benign  (91% confidence)
Omics:    Normal  (94% confidence)

Risk Score = 1 - (0.91 + 0.94) / 2 = 0.075
Risk Level = LOW RISK
Agreement  = Multimodal agreement on benign status
```

---

## Output Structure

```python
{
    "risk_score": 0.90,          # float: 0.0-1.0
    "risk_level": "HIGH RISK",   # string: LOW/MODERATE/HIGH
    "agreement": "Strong...",    # string: description
    "summary_text": "...",       # string: full clinical summary
    "img_cancer": True,          # bool: imaging finding
    "omics_cancer": True         # bool: molecular finding
}
```

---

## Clinical Summary Generation

The module generates a comprehensive text report with four sections:

### 1. Imaging Analysis Section
- Describes tissue classification result
- Mentions GradCAM findings for positive cases
- Includes confidence level

### 2. Genomic Analysis Section
- Reports molecular classification
- Lists top 3 dysregulated biomarkers (if cancer)
- References oncogenic pathways

### 3. Literature Evidence Section
- Counts valid PubMed publications retrieved
- Notes scientific consensus support

### 4. Clinical Recommendation Section
- Provides actionable guidance based on risk level
- Suggests next steps (biopsy, monitoring, etc.)

---

## Confidence Metrics

Additional metrics calculated by `calculate_confidence_metrics()`:

| Metric | Formula | Purpose |
|--------|---------|---------|
| Average Confidence | (img + omics) / 2 | Overall certainty |
| Max Confidence | max(img, omics) | Strongest signal |
| Min Confidence | min(img, omics) | Weakest signal |
| Agreement Strength | 1 - abs(img - omics) | How well modules agree |

---

## Integration with App

In `app.py` Tab 4 (Integrated Results):

```python
# Gather results from all modules
img_results = st.session_state.imaging_results
omics_results = st.session_state.omics_results
papers = st.session_state.literature_results or []

# Call integration module
summary = integration.generate_summary(
    img_results['prediction'], img_results['confidence'],
    omics_results['prediction'], omics_results['confidence'],
    omics_results['biomarkers'], papers
)

# Display unified results
st.metric("Risk Score", f"{summary['risk_score']:.1%}")
st.markdown(summary['summary_text'])
```

---

## Design Rationale

### Why 70% Weight for Discordant Results?
When imaging and omics disagree, we apply a 0.7 multiplier to the positive finding to:
1. **Reduce false positives**: Single-modality cancer detection is less reliable
2. **Flag for review**: Ensure discordant cases get additional testing
3. **Maintain safety**: Still elevates risk above baseline for clinical attention

### Why Average for Concordant Results?
When both modalities agree:
- **Positive concordance**: Average reflects combined evidence strength
- **Negative concordance**: Inverted average gives "benign confidence"

---

## File Location

```
breast_cancer_ai/
├── modules/
│   └── integration.py    ← This module
├── app.py                ← UI integration (Tab 4)
└── INTEGRATION_DOCUMENTATION.md  ← This file
```

---

## Version

- **Module Version**: 1.0
- **Last Updated**: February 2026
- **Dependencies**: None (pure Python logic)
