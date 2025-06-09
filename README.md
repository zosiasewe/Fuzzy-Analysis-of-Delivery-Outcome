# Fuzzy Analysis of Delivery Outcome
**Done in collaboration with @EvaBlackart.**
This project is a solution to selected tasks from the course **Fuzzy Data Analysis** under the supervision of **Dr. Robert Czabanski**, conducted in **Gliwice, 2024**.

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
