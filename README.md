# 📚 Aadarsh’s Statistics for Data Science

![Project Pipeline](https://pbs.twimg.com/media/E_l9hK-WEAQOzUV.png))

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



📌 Note: All topics will include formulas, explanation, Python code, and practical application wherever possible.

statistics-repo/
│── README.md
│── requirements.txt        # (optional, list of libraries: numpy, pandas, matplotlib, seaborn, scipy)
│
├── descriptive_stats/
│   ├── mean_median_mode.py
│   ├── variance_std.py
│   ├── skewness_kurtosis.py
│   └── summary_statistics.ipynb
│
├── probability/
│   ├── probability_basics.py
│   ├── conditional_probability.py
│   └── bayes_theorem.py
│
├── distributions/
│   ├── normal_distribution.py
│   ├── binomial_distribution.py
│   ├── poisson_distribution.py
│   ├── uniform_distribution.py
│   └── central_limit_theorem.py
│
├── hypothesis_tests/
│   ├── t_test.py
│   ├── z_test.py
│   ├── chi_square_test.py
│   ├── anova_test.py
│   └── non_parametric_tests.py
│
├── inferential_stats/
│   ├── confidence_intervals.py
│   └── sampling_methods.py
│
├── visualizations/
│   ├── boxplots_histograms.ipynb
│   ├── correlation_heatmap.ipynb
│   └── probability_curves.ipynb
│
├── real_world_projects/
│   ├── student_scores_analysis.ipynb
│   ├── health_data_analysis.ipynb
│   ├── titanic_survival_stats.ipynb
│   └── salary_distribution_analysis.ipynb
│
└── interview_prep/
    ├── stats_theory_questions.md
    ├── hypothesis_test_QA.md
    └── probability_QA.md

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
