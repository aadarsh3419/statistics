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

> 📌 **Note**: All topics will include formulas, explanation, Python code, and practical application wherever possible.

---
├── Measures of Central Tendency
├── Probability & Statistics Examples in Python
├── Python – Hypothesis Testing (t-test)
├── README.md
├── anova
├── basic
├── binom2
├── cc
├── chia square
├── ci
├── correlation
├── hypo  z test
├── mmm
├── normal distribution
├── one anova
├── p-value
├── paired t test
├── sd
├── sd and v
├── statistics-practice
    ├── README.md
    ├── mean_median_mode.py
    ├── t_test_example.py
    └── z_test_example.py
├── t -test
├── t test
├── t test g
├── t test two sample
├── t_test
├── the mean is same
├── to find ci
├── what do you mean by z test and t test
├── z- test
├── z- test k
├── z-score
└── z_score


/Measures of Central Tendency:
--------------------------------------------------------------------------------
1 | Question:  Find the mean, median, and mode of the dataset [4, 5, 6, 7, 5, 6, 6, 8, 9]  
2 | Answer:
3 | Mean = (4 + 5 + 6 + 7 + 5 + 6 + 6 + 8 + 9) / 9 = 56 / 9 = 6.22  
4 | Median = 6  
5 | Mode = 6 (since it appears most frequently)
6 | 


--------------------------------------------------------------------------------
/Probability & Statistics Examples in Python:
--------------------------------------------------------------------------------
 1 | """
 2 | Probability & Statistics Examples in Python
 3 | Author: Aadarsh Tiwari
 4 | Description: Collection of 10 examples demonstrating
 5 |              probability and statistics concepts in Python.
 6 | """
 7 | 
 8 | # 1. Probability of Specific Event (Basic Rule)
 9 | favorable_outcomes = 1
10 | total_outcomes = 6
11 | probability = favorable_outcomes / total_outcomes
12 | print("1. P(rolling a 4):", probability)
13 | 
14 | # 2. Complement Rule
15 | P_heads = 0.5
16 | P_not_heads = 1 - P_heads
17 | print("2. P(Not Heads):", P_not_heads)
18 | 
19 | # 3. Addition Rule for Mutually Exclusive Events
20 | P_A = 0.3
21 | P_B = 0.4
22 | P_A_or_B = P_A + P_B
23 | print("3. P(A or B) [Mutually Exclusive]:", P_A_or_B)
24 | 
25 | # 4. Addition Rule for Non-Mutually Exclusive Events
26 | P_A = 0.5
27 | P_B = 0.6
28 | P_A_and_B = 0.3
29 | P_A_or_B = P_A + P_B - P_A_and_B
30 | print("4. P(A or B) [Non-Mutually Exclusive]:", P_A_or_B)
31 | 
32 | # 5. Multiplication Rule for Independent Events
33 | P_A = 0.4
34 | P_B = 0.5
35 | P_A_and_B = P_A * P_B
36 | print("5. P(A and B) [Independent]:", P_A_and_B)
37 | 
38 | # 6. Multiplication Rule for Dependent Events
39 | P_A = 0.5
40 | P_B_given_A = 0.3
41 | P_A_and_B = P_A * P_B_given_A
42 | print("6. P(A and B) [Dependent]:", P_A_and_B)
43 | 
44 | # 7. Conditional Probability
45 | P_A_and_B = 0.2
46 | P_B = 0.4
47 | P_A_given_B = P_A_and_B / P_B
48 | print("7. P(A|B):", P_A_given_B)
49 | 
50 | # 8. Bayes’ Theorem
51 | P_D = 0.02          # P(Disease)
52 | P_not_D = 0.98      # P(No Disease)
53 | P_Pos_given_D = 0.90        # P(Positive | Disease)
54 | P_Pos_given_not_D = 0.05    # P(Positive | No Disease)
55 | 
56 | numerator = P_Pos_given_D * P_D
57 | denominator = (P_Pos_given_D * P_D) + (P_Pos_given_not_D * P_not_D)
58 | P_D_given_Pos = numerator / denominator
59 | print("8. P(Disease | Positive):", round(P_D_given_Pos, 4))
60 | 
61 | # 9. Binomial Probability
62 | from scipy.stats import binom
63 | n = 15
64 | p = 0.3
65 | k = 5
66 | probability = binom.pmf(k, n, p)
67 | print(f"9. P(X = {k} successes in {n} trials):", probability)
68 | 
69 | # 10. Normal Distribution Probability
70 | import scipy.stats as stats
71 | mean = 50
72 | std_dev = 10
73 | x = 60
74 | z_score = (x - mean) / std_dev
75 | P_less_than_x = stats.norm.cdf(z_score)
76 | print(f"10. P(X ≤ {x}):", P_less_than_x)
77 | 


--------------------------------------------------------------------------------
/Python – Hypothesis Testing (t-test):
--------------------------------------------------------------------------------
 1 | import scipy.stats as stats
 2 | 
 3 | # Sample data: before and after treatment sugar levels
 4 | before = [150, 160, 155, 145, 170]
 5 | after = [140, 150, 145, 138, 160]
 6 | 
 7 | t_stat, p_value = stats.ttest_rel(before, after)
 8 | 
 9 | print("t-statistic:", t_stat)
10 | print("p-value:", p_value)
11 | 
12 | if p_value < 0.05:
13 |     print("Significant difference after treatment.")
14 | else:
15 |     print("No significant difference.")
16 | 


--------------------------------------------------------------------------------
/README.md:
--------------------------------------------------------------------------------
 1 | # 📚 Aadarsh’s Statistics for Data Science
 2 | 
 3 | ![Project Pipeline](https://pbs.twimg.com/media/E_l9hK-WEAQOzUV.png))
 4 | 
 5 | Welcome to my **Statistics Repository**, where I’m mastering all the essential concepts from beginner to advanced — with real-life examples, Python code, formulas, and applications in **Data Science** and **Machine Learning**.
 6 | 
 7 | This repo is part of my self-learning journey to become a **confident and job-ready Data Scientist**.
 8 | 
 9 | ---
10 | 
11 | ## 📌 What You’ll Find Here
12 | 
13 | ✅ Complete A to Z Statistics Concepts  
14 | ✅ Real-world explanations + formulas  
15 | ✅ Python code using `scipy`, `numpy`, `pandas`  
16 | ✅ Practice questions + solved examples  
17 | ✅ Data Science interview preparation  
18 | ✅ Revision-friendly layout + cheat sheets (coming soon)  
19 | 
20 | ---
21 | 
22 | ## 🎯 My Statistics Learning Map
23 | 
24 | | ✅ | Topic |
25 | |----|-------|
26 | | ✔️ | Central Tendency (Mean, Median, Mode) |
27 | | ✔️ | Measure of Dispersion (Range, Variance, SD, IQR) |
28 | | ✔️ | Probability Basics |
29 | | ✔️ | Types of Data (Qualitative, Quantitative) |
30 | | ✔️ | Random Variables (Discrete & Continuous) |
31 | | ✔️ | Probability Mass Function (PMF) |
32 | | ✔️ | Probability Density Function (PDF) |
33 | | ✔️ | Cumulative Distribution Function (CDF) |
34 | | ✔️ | Uniform Distribution |
35 | | ✔️ | Binomial Distribution |
36 | | ✔️ | Normal Distribution |
37 | | ✔️ | Poisson Distribution |
38 | | ✔️ | Geometric Distribution |
39 | | ✔️ | Hypothesis Testing |
40 | | ✔️ | Z-Test |
41 | | ✔️ | T-Test |
42 | | ✔️ | Chi-Square Test |
43 | | ✔️ | ANOVA Test (F-test) |
44 | | ✔️ | Central Limit Theorem |
45 | | ✔️ | Confidence Intervals |
46 | | ✔️ | Bayes Theorem |
47 | | ✔️ | Correlation & Covariance |
48 | | ✔️ | Skewness & Kurtosis |
49 | | ✔️ | Outliers & Boxplots |
50 | | ✔️ | Descriptive vs Inferential Statistics |
51 | | ✔️ | Sampling Methods |
52 | | ✔️ | P-value & Significance Level |
53 | | ✔️ | Real-Life Projects using Statistics |
54 | | ✔️ | Interview Q&A (coming soon) |
55 | 
56 | > 📌 **Note**: All topics will include formulas, explanation, Python code, and practical application wherever possible.
57 | 
58 | ---
59 | 
60 | ## 🧑‍💻 Sample Python Code – Bayes Theorem
61 | 
62 | ```python
63 | # Given values
64 | P_D = 0.02          # P(Disease)
65 | P_not_D = 0.98      # P(No Disease)
66 | P_Pos_given_D = 0.90        # P(Positive | Disease)
67 | P_Pos_given_not_D = 0.05    # P(Positive | No Disease)
68 | 
69 | # Bayes Theorem
70 | numerator = P_Pos_given_D * P_D
71 | denominator = (P_Pos_given_D * P_D) + (P_Pos_given_not_D * P_not_D)
72 | P_D_given_Pos = numerator / denominator
73 | 
74 | print("P(Disease | Positive):", round(P_D_given_Pos, 4))
75 | 


--------------------------------------------------------------------------------
/anova:
--------------------------------------------------------------------------------
 1 | **Topic:** ANOVA  
 2 | **Question:** Test if the means of 3 groups are significantly different:  
 3 | Group A: [5, 7, 6]  
 4 | Group B: [8, 9, 7]  
 5 | Group C: [6, 5, 7]  
 6 | 
 7 | **Answer (Python code used):**
 8 | ```python
 9 | from scipy.stats import f_oneway
10 | group_a = [5, 7, 6]
11 | group_b = [8, 9, 7]
12 | group_c = [6, 5, 7]
13 | f_stat, p_value = f_oneway(group_a, group_b, group_c)
14 | print(f"F = {f_stat}, p = {p_value}")
15 | 


--------------------------------------------------------------------------------
/basic:
--------------------------------------------------------------------------------
 1 | """
 2 | Topic: Central Tendency – Mean, Median, Mode
 3 | 
 4 | Self-Practice Notes
 5 | 
 6 | I took a small dataset of student marks and tried to calculate mean, median, and mode manually and using Python to reinforce the concept.
 7 | """
 8 | 
 9 | import numpy as np
10 | from scipy import stats
11 | 
12 | # Sample data: Marks of 10 students in a test
13 | marks = [45, 50, 55, 60, 65, 55, 70, 75, 55, 80]
14 | 
15 | # Mean
16 | mean_marks = np.mean(marks)
17 | print(f"Mean: {mean_marks}")
18 | 
19 | # Median
20 | median_marks = np.median(marks)
21 | print(f"Median: {median_marks}")
22 | 
23 | # Mode
24 | mode_marks = stats.mode(marks, keepdims=False)
25 | print(f"Mode: {mode_marks.mode}, Count: {mode_marks.count}")
26 | 


--------------------------------------------------------------------------------
/binom2:
--------------------------------------------------------------------------------
1 | from scipy.stats import binom
2 | 
3 | n = 15
4 | p = 0.3
5 | k = 5
6 | 
7 | probability = binom.pmf(k, n, p)
8 | print(f"P(X = {k} successes in {n} trials):", probability)
9 | 


--------------------------------------------------------------------------------
/cc:
--------------------------------------------------------------------------------
 1 | """
 2 | Topic: Pearson Correlation Coefficient
 3 | 
 4 | I checked the relationship between hours studied and marks obtained.
 5 | """
 6 | 
 7 | import numpy as np
 8 | from scipy.stats import pearsonr
 9 | 
10 | # Data
11 | hours = [1, 2, 3, 4, 5, 6]
12 | marks = [50, 55, 65, 70, 75, 85]
13 | 
14 | # Pearson correlation
15 | corr, p_value = pearsonr(hours, marks)
16 | 
17 | print(f"Pearson Correlation Coefficient: {corr:.2f}")
18 | print(f"P-value: {p_value:.4f}")
19 | 
20 | # Interpretation
21 | if p_value < 0.05:
22 |     print("Strong linear correlation exists.")
23 | else:
24 |     print("No significant correlation.")
25 | 


--------------------------------------------------------------------------------
/chia square:
--------------------------------------------------------------------------------
 1 | import pandas as pd
 2 | from scipy.stats import chi2_contingency
 3 | 
 4 | # Sample dataset
 5 | data = pd.DataFrame({
 6 |     "Gender": ["Male", "Male", "Female", "Female", "Male", "Female", "Male", "Female"],
 7 |     "Preference": ["Tea", "Coffee", "Tea", "Tea", "Coffee", "Coffee", "Tea", "Coffee"]
 8 | })
 9 | 
10 | # Create contingency table
11 | contingency_table = pd.crosstab(data["Gender"], data["Preference"])
12 | 
13 | # Chi-square test
14 | chi2, p, dof, expected = chi2_contingency(contingency_table)
15 | 
16 | print("Chi-square Statistic:", chi2)
17 | print("p-value:", p)
18 | print("Degrees of Freedom:", dof)
19 | print("Expected Table:\n", expected)
20 | 
21 | if p < 0.05:
22 |     print("Conclusion: Significant relationship between Gender and Preference")
23 | else:
24 |     print("Conclusion: No significant relationship between Gender and Preference")
25 | 


--------------------------------------------------------------------------------
/ci:
--------------------------------------------------------------------------------
 1 | """
 2 | Topic: 95% Confidence Interval for a Sample Mean
 3 | 
 4 | Calculated CI to estimate where the true population mean may lie based on sample.
 5 | """
 6 | 
 7 | import numpy as np
 8 | from scipy.stats import t
 9 | 
10 | # Sample data
11 | sample = [70, 72, 68, 74, 69, 71, 73, 75, 72, 70]
12 | n = len(sample)
13 | mean = np.mean(sample)
14 | s = np.std(sample, ddof=1)
15 | 
16 | # 95% confidence interval
17 | confidence = 0.95
18 | alpha = 1 - confidence
19 | df = n - 1
20 | t_crit = t.ppf(1 - alpha/2, df)
21 | se = s / np.sqrt(n)
22 | margin = t_crit * se
23 | 
24 | ci_lower = mean - margin
25 | ci_upper = mean + margin
26 | 
27 | print(f"Sample Mean: {mean:.2f}")
28 | print(f"95% Confidence Interval: ({ci_lower:.2f}, {ci_upper:.2f})")
29 | 


--------------------------------------------------------------------------------
/correlation:
--------------------------------------------------------------------------------
1 | x = [1, 2, 3, 4, 5]
2 | y = [2, 4, 6, 8, 10]
3 | 
4 | correlation = np.corrcoef(x, y)
5 | print("Correlation Matrix:\n", correlation)
6 | 


--------------------------------------------------------------------------------
/hypo  z test:
--------------------------------------------------------------------------------
 1 | import numpy as np
 2 | from statsmodels.stats.weightstats import ztest
 3 | 
 4 | # Sample data
 5 | sample_data = [2.3, 2.5, 2.7, 2.8, 2.6, 2.4, 2.9, 2.5, 2.6, 2.7]
 6 | 
 7 | # Perform one-sample z-test (test mean = 2.5)
 8 | z_stat, p_value = ztest(sample_data, value=2.5)
 9 | 
10 | print("Z-statistic:", z_stat)
11 | print("p-value:", p_value)
12 | 
13 | if p_value < 0.05:
14 |     print("Conclusion: Mean is significantly different from 2.5")
15 | else:
16 |     print("Conclusion: Mean is not significantly different from 2.5")
17 | 


--------------------------------------------------------------------------------
/mmm:
--------------------------------------------------------------------------------
 1 | import numpy as np
 2 | from scipy import stats
 3 | 
 4 | data = [12, 15, 17, 15, 19, 21, 15]
 5 | 
 6 | mean = np.mean(data)
 7 | median = np.median(data)
 8 | mode = stats.mode(data)
 9 | 
10 | print("Mean:", mean)
11 | print("Median:", median)
12 | print("Mode:", mode.mode[0])
13 | 


--------------------------------------------------------------------------------
/normal distribution:
--------------------------------------------------------------------------------
 1 | import scipy.stats as stats
 2 | 
 3 | mean = 50
 4 | std_dev = 10
 5 | x = 60
 6 | 
 7 | z_score = (x - mean) / std_dev
 8 | P_less_than_x = stats.norm.cdf(z_score)
 9 | print(f"P(X ≤ {x}):", P_less_than_x)
10 | 


--------------------------------------------------------------------------------
/one anova:
--------------------------------------------------------------------------------
 1 | from scipy.stats import f_oneway
 2 | 
 3 | # Sample data: Exam scores for 3 different teaching methods
 4 | method_A = [85, 88, 90, 93, 95]
 5 | method_B = [78, 82, 84, 80, 79]
 6 | method_C = [92, 94, 96, 91, 90]
 7 | 
 8 | # Perform ANOVA
 9 | f_stat, p_value = f_oneway(method_A, method_B, method_C)
10 | 
11 | print("F-statistic:", f_stat)
12 | print("p-value:", p_value)
13 | 
14 | if p_value < 0.05:
15 |     print("Conclusion: At least one teaching method differs significantly.")
16 | else:
17 |     print("Conclusion: No significant difference between methods.")
18 | 


--------------------------------------------------------------------------------
/p-value:
--------------------------------------------------------------------------------
 1 | # Answer:
 2 | # A high p-value (> 0.05) means we fail to reject the null hypothesis.
 3 | # It means there is not enough evidence to prove something changed.
 4 | # Example:
 5 | p_value = 0.72
 6 | if p_value > 0.05:
 7 |     print("Fail to reject the null hypothesis.")
 8 | else:
 9 |     print("Reject the null hypothesis.")
10 | 


--------------------------------------------------------------------------------
/paired t test:
--------------------------------------------------------------------------------
 1 | from scipy.stats import ttest_rel
 2 | 
 3 | # Before and After scores of same students
 4 | before = [70, 75, 80, 85, 90]
 5 | after = [72, 78, 82, 88, 92]
 6 | 
 7 | # Perform paired t-test
 8 | t_stat, p_value = ttest_rel(before, after)
 9 | 
10 | print("T-statistic:", t_stat)
11 | print("p-value:", p_value)
12 | 
13 | if p_value < 0.05:
14 |     print("Conclusion: Significant improvement after intervention.")
15 | else:
16 |     print("Conclusion: No significant improvement after intervention.")
17 | 


--------------------------------------------------------------------------------
/sd:
--------------------------------------------------------------------------------
 1 | # Answer:
 2 | # Standard Deviation tells us how spread out the data is from the mean.
 3 | # Variance is just the square of Standard Deviation.
 4 | # For example:
 5 | import numpy as np
 6 | data = [10, 12, 14, 16, 18]
 7 | std = np.std(data)
 8 | var = np.var(data)
 9 | print("Standard Deviation:", std)
10 | print("Variance:", var)
11 | 


--------------------------------------------------------------------------------
/sd and v:
--------------------------------------------------------------------------------
1 | data = [10, 20, 30, 40, 50]
2 | 
3 | std_dev = np.std(data)
4 | variance = np.var(data)
5 | 
6 | print("Standard Deviation:", std_dev)
7 | print("Variance:", variance)
8 | 


--------------------------------------------------------------------------------
/statistics-practice/README.md:
--------------------------------------------------------------------------------
https://raw.githubusercontent.com/aadarsh3419/statistics/14649425254dbb7ba3e42c8e191e3a5723cfcc37/statistics-practice/README.md


--------------------------------------------------------------------------------
/statistics-practice/mean_median_mode.py:
--------------------------------------------------------------------------------
https://raw.githubusercontent.com/aadarsh3419/statistics/14649425254dbb7ba3e42c8e191e3a5723cfcc37/statistics-practice/mean_median_mode.py


--------------------------------------------------------------------------------
/statistics-practice/t_test_example.py:
--------------------------------------------------------------------------------
https://raw.githubusercontent.com/aadarsh3419/statistics/14649425254dbb7ba3e42c8e191e3a5723cfcc37/statistics-practice/t_test_example.py


--------------------------------------------------------------------------------
/statistics-practice/z_test_example.py:
--------------------------------------------------------------------------------
https://raw.githubusercontent.com/aadarsh3419/statistics/14649425254dbb7ba3e42c8e191e3a5723cfcc37/statistics-practice/z_test_example.py


--------------------------------------------------------------------------------
/t -test:
--------------------------------------------------------------------------------
 1 | **Topic:** T-Test  
 2 | **Question:** A sample of 20 students has an average score of 75. The population mean is 80, and standard deviation is 10. Is there a significant difference?  
 3 | **Answer:**  
 4 | n = 20  
 5 | x̄ = 75, μ = 80, s = 10  
 6 | t = (x̄ - μ) / (s / √n) = (75 - 80) / (10 / √20) = -2.24  
 7 | Compare with t-critical value (α = 0.05): ±2.093  
 8 | Since -2.24 < -2.093 → Reject null hypothesis  
 9 | There is significant difference.
10 | 


--------------------------------------------------------------------------------
/t test:
--------------------------------------------------------------------------------
 1 | # T-Test 
 2 | 
 3 | 
 4 | import numpy as np
 5 | 
 6 | # Given Data
 7 | sample = [19.2, 21.1, 20.3, 18.7, 20.1, 19.9, 20.5, 19.5, 20.4]
 8 | n = 9
 9 | population_mean = 20
10 | t_value = 1.96
11 | df = n - 1
12 | 
13 | # Step 1: Mean and SD
14 | m = np.mean(sample)
15 | s = np.std(sample, ddof=1)
16 | 
17 | # Step 2: T-Test Calculation
18 | se = s / np.sqrt(n)
19 | t_stat = (m - population_mean) / se
20 | 
21 | print(f"Sample Mean: {m:.2f}")
22 | print(f"Standard Deviation: {s:.2f}")
23 | print(f"T-Statistic: {t_stat:.3f}")
24 | print(f"Degrees of Freedom: {df}")
25 | 


--------------------------------------------------------------------------------
/t test g:
--------------------------------------------------------------------------------
 1 | # Q1. T-test to compare sample mean and population mean
 2 | import numpy as np
 3 | from scipy import stats
 4 | 
 5 | # Sample details
 6 | sample_mean = 15.4
 7 | population_mean = 16
 8 | sample_std_dev = 2.1
 9 | sample_size = 20
10 | 
11 | # Step 1: Calculate Standard Error (SE)
12 | se = sample_std_dev / np.sqrt(sample_size)
13 | 
14 | # Step 2: Calculate t-statistic
15 | t_stat = (sample_mean - population_mean) / se
16 | 
17 | # Step 3: Degrees of freedom
18 | df = sample_size - 1
19 | 
20 | # Step 4: Calculate p-value
21 | p_value = stats.t.sf(np.abs(t_stat), df) * 2  # two-tailed test
22 | 
23 | print("T-Statistic:", t_stat)
24 | print("Degrees of Freedom:", df)
25 | print("P-Value:", p_value)
26 | 
27 | # 📌 Conclusion
28 | if p_value < 0.05:
29 |     print("→ Reject null hypothesis: Sample mean is significantly different from population mean.")
30 | else:
31 |     print("→ Fail to reject null: No significant difference found.")
32 | 


--------------------------------------------------------------------------------
/t test two sample:
--------------------------------------------------------------------------------
 1 | from scipy.stats import ttest_ind
 2 | 
 3 | # Sample data: Scores of two groups
 4 | group_A = [85, 88, 90, 93, 95]
 5 | group_B = [78, 82, 84, 80, 79]
 6 | 
 7 | # Perform two-sample t-test
 8 | t_stat, p_value = ttest_ind(group_A, group_B)
 9 | 
10 | print("T-statistic:", t_stat)
11 | print("p-value:", p_value)
12 | 
13 | if p_value < 0.05:
14 |     print("Conclusion: Significant difference between the two groups.")
15 | else:
16 |     print("Conclusion: No significant difference between the two groups.")
17 | 


--------------------------------------------------------------------------------
/t_test:
--------------------------------------------------------------------------------
1 | from scipy.stats import ttest_1samp
2 | 
3 | sample = [22, 24, 20, 23, 21, 19]
4 | population_mean = 20
5 | 
6 | t_stat, p_value = ttest_1samp(sample, population_mean)
7 | print("T-statistic:", t_stat)
8 | print("P-value:", p_value)
9 | 


--------------------------------------------------------------------------------
/the mean is same:
--------------------------------------------------------------------------------
 1 | import numpy as np  
 2 | import matplotlib.pyplot as plt  
 3 | 
 4 | # Generate population data  
 5 | population = np.random.normal(loc=50, scale=15, size=10000)  
 6 | 
 7 | # Draw multiple sample means  
 8 | sample_means = [np.mean(np.random.choice(population, 50)) for _ in range(1000)]  
 9 | 
10 | # Plot sample means  
11 | plt.hist(sample_means, bins=30, color='skyblue', edgecolor='black')  
12 | plt.title("Central Limit Theorem in Action")  
13 | plt.xlabel("Sample Mean")  
14 | plt.ylabel("Frequency")  
15 | plt.show()
16 | 


--------------------------------------------------------------------------------
/to find ci:
--------------------------------------------------------------------------------
 1 | import numpy as np
 2 | 
 3 | n = 36
 4 | mu = 52
 5 | sd = 8
 6 | z_alpha_2 = 1.96
 7 | 
 8 | se = sd / np.sqrt(n)
 9 | moe = z_alpha_2 * se
10 | 
11 | ci_lower = mu - moe
12 | ci_upper = mu + moe
13 | 
14 | print("Confidence Interval is: (", round(ci_lower, 2), ",", round(ci_upper, 2), ")")
15 | print("Margin of Error is:", round(moe, 2))
16 | 


--------------------------------------------------------------------------------
/what do you mean by z test and t test:
--------------------------------------------------------------------------------
 1 | # Answer:
 2 | # Use Z-test when population SD is known or n > 30.
 3 | # Use T-test when population SD is unknown and n < 30.
 4 | # Example:
 5 | # One-sample t-test (manually checking if sample mean ≠ known mean)
 6 | import scipy.stats as stats
 7 | sample = [15, 17, 14, 16, 18]
 8 | t_stat, p_value = stats.ttest_1samp(sample, 16)
 9 | print("T-Statistic:", t_stat)
10 | print("P-value:", p_value)
11 | 


--------------------------------------------------------------------------------
/z- test:
--------------------------------------------------------------------------------
 1 | """
 2 | Topic: Z-Test for Population Mean
 3 | 
 4 | In this problem, I checked if the sample mean significantly differs from the known population mean.
 5 | """
 6 | 
 7 | import numpy as np
 8 | from scipy.stats import norm
 9 | 
10 | # Sample details
11 | sample_mean = 78
12 | population_mean = 75
13 | std_dev = 10
14 | n = 36  # sample size
15 | 
16 | # Standard Error
17 | se = std_dev / np.sqrt(n)
18 | 
19 | # Z-value
20 | z = (sample_mean - population_mean) / se
21 | p_value = 1 - norm.cdf(z)
22 | 
23 | print(f"Z-value: {z:.2f}")
24 | print(f"P-value: {p_value:.4f}")
25 | 
26 | # Interpretation
27 | if p_value < 0.05:
28 |     print("Reject Null Hypothesis – Significant difference.")
29 | else:
30 |     print("Fail to Reject Null – No significant difference.")
31 | 


--------------------------------------------------------------------------------
/z- test k:
--------------------------------------------------------------------------------
 1 |  Z-test to compare sample mean and population mean (known std)
 2 | import numpy as np
 3 | from scipy import stats
 4 | 
 5 | 
 6 | sample_mean = 49.2
 7 | population_mean = 50
 8 | std_dev = 1.5
 9 | sample_size = 40
10 | 
11 |  Calculate Standard Error (SE)
12 | se = std_dev / np.sqrt(sample_size)
13 | 
14 |  Calculate Z-score
15 | z_score = (sample_mean - population_mean) / se
16 | 
17 |  Calculate p-value
18 | p_value = stats.norm.sf(abs(z_score)) * 2  # two-tailed
19 | 
20 | print("Z-Score:", z_score)
21 | print("P-Value:", p_value)
22 | 
23 |  Conclusion
24 | if p_value < 0.05:
25 |     print("→ Reject null hypothesis: Output mean is significantly different.")
26 | else:
27 |     print("→ Fail to reject null: No significant difference.")
28 | 


--------------------------------------------------------------------------------
/z-score:
--------------------------------------------------------------------------------
1 | **Topic:** Z-Score  
2 | **Question:** Find the Z-score for x = 85, when mean μ = 70 and standard deviation σ = 10  
3 | **Answer:**  
4 | Z = (x - μ) / σ = (85 - 70) / 10 = 1.5
5 | 


--------------------------------------------------------------------------------
/z_score:
--------------------------------------------------------------------------------
1 | data = [80, 85, 90, 75, 70, 95]
2 | mean = np.mean(data)
3 | std = np.std(data)
4 | 
5 | z_scores = [(x - mean) / std for x in data]
6 | print("Z-Scores:", z_scores)

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
