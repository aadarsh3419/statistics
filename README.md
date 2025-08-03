# 📚 Aadarsh’s Statistics for Data Science

![Project Pipeline](https://www.google.com/imgres?q=statistics%20animation%20images%20moving&imgurl=https%3A%2F%2Fstatic.vecteezy.com%2Fsystem%2Fresources%2Fthumbnails%2F026%2F602%2F755%2Flarge%2Fbusiness-analysis-and-statistics-moving-male-entrepreneur-studies-digital-graphs-and-company-charts-or-financial-report-character-develops-strategy-for-success-isometric-graphic-animated-cartoon-video.jpg&imgrefurl=https%3A%2F%2Fwww.vecteezy.com%2Ffree-videos%2Fanimated-statistics&docid=xyf_LI9rits8uM&tbnid=oxrykN4f50tp3M&vet=12ahUKEwi58Onmhu-OAxVHTGwGHeOmHDoQM3oECBoQAA..i&w=800&h=450&hcb=2&ved=2ahUKEwi58Onmhu-OAxVHTGwGHeOmHDoQM3oECBoQAA))

Welcome to my **Statistics Repository**, where I’m mastering all the essential concepts from beginner to advanced — with real-life examples, Python code, formulas, and applications in **Data Science** and **Machine Learning**.

This repo is part of my self-learning journey to become a **confident and job-ready Data Scientist**.

---

## 📌 What You’ll Find Here

✅ Complete A to Z Statistics Concepts  
✅ Real-world explanations + formulas  
✅ Python code using `scipy`, `numpy`, `pandas`  
✅ Practice questions + solved examples  
✅ Data Science interview preparation  
✅ Revision-friendly layout + cheat sheets (coming soon)  

---

## 🎯 My Statistics Learning Map

| ✅ | Topic |
|----|-------|
| ✔️ | Central Tendency (Mean, Median, Mode) |
| ✔️ | Measure of Dispersion (Range, Variance, SD, IQR) |
| ✔️ | Probability Basics |
| ✔️ | Types of Data (Qualitative, Quantitative) |
| ✔️ | Random Variables (Discrete & Continuous) |
| ✔️ | Probability Mass Function (PMF) |
| ✔️ | Probability Density Function (PDF) |
| ✔️ | Cumulative Distribution Function (CDF) |
| ✔️ | Uniform Distribution |
| ✔️ | Binomial Distribution |
| ✔️ | Normal Distribution |
| ✔️ | Poisson Distribution |
| ✔️ | Geometric Distribution |
| ✔️ | Hypothesis Testing |
| ✔️ | Z-Test |
| ✔️ | T-Test |
| ✔️ | Chi-Square Test |
| ✔️ | ANOVA Test (F-test) |
| ✔️ | Central Limit Theorem |
| ✔️ | Confidence Intervals |
| ✔️ | Bayes Theorem |
| ✔️ | Correlation & Covariance |
| ✔️ | Skewness & Kurtosis |
| ✔️ | Outliers & Boxplots |
| ✔️ | Descriptive vs Inferential Statistics |
| ✔️ | Sampling Methods |
| ✔️ | P-value & Significance Level |
| ✔️ | Real-Life Projects using Statistics |
| ✔️ | Interview Q&A (coming soon) |

> 📌 **Note**: All topics will include formulas, explanation, Python code, and practical application wherever possible.

---

## 🧑‍💻 Sample Python Code – Bayes Theorem

```python
# Given values
P_D = 0.02          # P(Disease)
P_not_D = 0.98      # P(No Disease)
P_Pos_given_D = 0.90        # P(Positive | Disease)
P_Pos_given_not_D = 0.05    # P(Positive | No Disease)

# Bayes Theorem
numerator = P_Pos_given_D * P_D
denominator = (P_Pos_given_D * P_D) + (P_Pos_given_not_D * P_not_D)
P_D_given_Pos = numerator / denominator

print("P(Disease | Positive):", round(P_D_given_Pos, 4))
