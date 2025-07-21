"""
Topic: Pearson Correlation Coefficient

I checked the relationship between hours studied and marks obtained.
"""

import numpy as np
from scipy.stats import pearsonr

# Data
hours = [1, 2, 3, 4, 5, 6]
marks = [50, 55, 65, 70, 75, 85]

# Pearson correlation
corr, p_value = pearsonr(hours, marks)

print(f"Pearson Correlation Coefficient: {corr:.2f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
if p_value < 0.05:
    print("Strong linear correlation exists.")
else:
    print("No significant correlation.")
