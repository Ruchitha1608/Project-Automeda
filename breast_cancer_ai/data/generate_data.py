"""
Generate sample omics data for demo
"""
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Known breast cancer biomarkers
known_genes = ['BRCA1', 'BRCA2', 'TP53', 'HER2', 'ESR1', 'PGR', 'MYC', 'PIK3CA', 'PTEN', 'EGFR']

# Additional random genes
random_genes = [f'Gene{i}' for i in range(1, 191)]

# All gene names
all_genes = known_genes + random_genes

# Generate 100 samples (60 cancer, 40 normal)
n_samples = 100
n_cancer = 60
n_normal = 40

data = {}

# Generate expression data for each gene
for gene in all_genes:
    if gene in known_genes:
        # Known biomarkers have differential expression
        normal_expr = np.random.randn(n_normal) * 1.5 + 5.0
        cancer_expr = np.random.randn(n_cancer) * 1.8 + 7.5  # Higher in cancer
        data[gene] = np.concatenate([cancer_expr, normal_expr])
    else:
        # Random genes have similar expression
        data[gene] = np.random.randn(n_samples) * 1.2 + 5.5

# Add labels
labels = ['Cancer'] * n_cancer + ['Normal'] * n_normal
data['label'] = labels

# Create DataFrame
df = pd.DataFrame(data)

# Shuffle rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
df.to_csv('/Users/work/major proj/breast_cancer_ai/data/sample_omics.csv', index=False)

print(f"Generated sample omics data: {df.shape}")
print(f"Cancer samples: {sum(df['label'] == 'Cancer')}")
print(f"Normal samples: {sum(df['label'] == 'Normal')}")
print(f"Genes: {len(all_genes)}")
