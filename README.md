# Fuzzy Analysis of Delivery Outcome
**Done in collaboration with @EvaBlackart.**
This project is a solution to selected tasks from the course **Fuzzy Data Analysis** under the supervision of **Dr. Robert Czabanski**, conducted in **Gliwice, 2025**.

##  Project Overview

The project focuses on the **retrospective assessment of fetal state** using fuzzy logic models based on delivery outcome attributes: **Apgar score (AP)**, **Birth weight (BW)**, and **Umbilical cord pH (PH)**. 

A **Takagi-Sugeno-Kang (TSK)** fuzzy inference system was implemented to assess fetal conditions as:
- **Normal**
- **Suspicious**
- **Abnormal**

The project explores fuzzy rule creation, optimization via grid search, and comparison to pH-based assessments.

## Completed Tasks

###  Membership Function Definition
- Designed fuzzy membership functions for AP, BW, and PH.
- Implemented trapezoidal membership functions as per statistical ranges provided.
- Established fuzzy rules for the TSK model using combinations of the outcome attributes.

![image](https://github.com/user-attachments/assets/7c44dbe6-9e86-494c-aa05-61d107e741d7)
![image](https://github.com/user-attachments/assets/6835330f-ba22-48d3-87ff-eb3b9f03df95)
![image](https://github.com/user-attachments/assets/8a701a5a-847d-4242-8104-cf1770e631da)

###  Parameter Optimization via Grid Search
- Tuned fuzzy model parameters:
  - Singleton values `p(i)` for suspicious outcomes.
  - Threshold ∆ for binary classification.
- Performed **grid search** over the ranges `[-0.5, 0.5]` with a step of `0.25`.
- Evaluated performance using **5-fold cross-validation**.
- Measured classification quality using the **mean G-measure** based on comparison with pH-based outcome classification.

**Results Obtained:**
    Best from fold 1 → G-measure = 0.9354, p_value = -0.5, delta = -0.5
    G-measure on the test data = 0.9428
    Best from fold 2 → G-measure = 0.9276, p_value = -0.5, delta = -0.5
    G-measure on the test data = 0.9636
    Best from fold 3 → G-measure = 0.9476, p_value = -0.5, delta = -0.5
    G-measure on the test data = 0.8660
    Best from fold 4 → G-measure = 0.9189, p_value = -0.5, delta = -0.5
    G-measure on the test data = 1.0000
    Best from fold 5 → G-measure = 0.9524, p_value = -0.5, delta = -0.5
    G-measure on the test data = 0.8864
    Results across all folds: Median G-measure = 0.9428
    Final G - measure output (median) : 0.9428
    
###  Parameter Optimization via Evolutionary Strategy
- Tuned fuzzy model parameters:
  - p_i for each suspicious rule
  - delta value
- Performed **Evolutionary Strategy** over the ranges `[-0.5, 0.5]` of initial parameters 
- Evaluated performance using **5-fold cross-validation**.
- Measured classification quality using the **mean G-measure**

**Results Obtained:**
  Selection mode: mi_plus_lambda
  Number of suspicious rules: 4
  Total parameters to optimize: 5
  Population size μ = 100, Offspring size λ = 500
  
  === Generation 1 ===
  Best fitness: 0.9428
  Offspring best fitness: 0.9428
  
  === Generation 2 ===
  Best fitness: 0.9428
  
  Converged after 2 generations
  Final fitness delta: 0.00e+00
  
   Best Solution 
  delta : -0.2690
  
  p(i) values for suspicious rules : 
  Rule (0.5, 0.5, 0.5): p(0) = 0.2192
  Rule (0.5, -1.0, 0.5): p(1) = -0.3355
  Rule (-1.0, 0.5, 0.5): p(2) = -0.2341
  Rule (0.5, 0.5, -1.0): p(3) = 0.4376
  
  Signals with high informativeness |y0| > 0.5 : 166


##  Technologies Used
- Python
- NumPy
- scikit-learn
- matplotlib (for visualizations)

##  Dataset
The dataset contains fetal monitoring outcomes and corresponding delivery attributes, provided by the course instructor.

##  Next Steps
-  Task 3: Use evolutionary strategies to further optimize `p(i)` and ∆.
-  Analyze informativeness of signals with |y₀| > 0.5 based on best model configuration.

---

**Author:** Zofia Sewerynska; Ewa Radwan   
**Course:** Fuzzy Data Analysis – 2024 5
**Supervisor:** Robert Czabanski, PhD, DSc  
