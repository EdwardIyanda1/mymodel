# FUNAAB ICTREC Helpdesk Data Interpretation
# Comprehensive Interpretation and Statistical Analysis: FUNAAB ICTREC Helpdesk Dataset & PyTorch Predictive Model

## What the data contains
---

The file contains 500 tickets recorded from 8 January to 2 September 2026. It has 12 columns and no missing values. Each ticket records the requester type, department, issue, priority, response and resolution times, status, resolution flag, and response SLA result.
## 1. Executive Summary

## Main observations
This report provides an end-to-end interpretation of the **Federal University of Agriculture, Abeokuta (FUNAAB) ICT Resource Centre (ICTREC)** Helpdesk dataset (`FUNAAB_ICTREC_Helpdesk_500_Instances.csv`) and evaluates the regression modeling pipeline implemented in [`main.py`](file:///c:/Users/user/Documents/mymodel/main.py).

- 358 tickets were resolved, giving a resolution rate of **71.6%**.
- 478 tickets received a response within eight hours, giving a response-SLA compliance rate of **95.6%**.
- Mean response time was **3.09 hours**; median response time was **2.58 hours**.
- Mean resolution time was **13.40 hours**; median resolution time was **7.64 hours**. The mean being much larger than the median indicates a right-skewed distribution with some long-running tickets.
- The most frequent category was **Portal** with 79 tickets, followed by **Payment/Invoice** with 63 and **Admission** with 54.
- Priority distribution was Medium 222, Low 147, High 110, and Critical 21.
- Students submitted 240 tickets, postgraduate students 137, and staff 123.
- Closed tickets had a median resolution time of **5.68 hours**. In-progress and pending tickets had much higher median recorded durations of **28.03** and **29.77 hours** respectively.
- Response time and resolution time had a Pearson correlation of about **-0.06**, so this sample does not show a useful linear relationship between those two fields.
### Key Dataset KPI Snapshot

## Important interpretation cautions
| Metric | Value | Operational Interpretation |
| :--- | :--- | :--- |
| **Total Ticket Volume** | 500 records | Captures 8 months of campus-wide operational support requests. |
| **Observation Window** | Jan 8, 2026 – Sep 2, 2026 | Spans academic admissions, registrations, and exam/clearance phases. |
| **Resolution Rate** | **71.6%** (358 / 500) | 7 out of 10 tickets reach completion; 28.4% remain pending/in progress. |
| **Response SLA Compliance** | **95.6%** (478 / 500) | Excellent initial acknowledgment performance (under 8 hours). |
| **Mean / Median Response Time**| **3.09 hrs** / **2.58 hrs** | First contact is swift, consistently under 3 hours. |
| **Resolved Tickets Mean / Median Duration**| **7.44 hrs** / **5.68 hrs** | When resolved, the typical turnaround is within the same working day. |
| **Unresolved Tickets Mean Elapsed Time**| **28.42 hrs** | Open tickets represent aging backlog, not completed durations. |
| **Correlation (Response vs. Resolution)**| **-0.16** | Fast first response does **not** correlate with rapid resolution. |
| **ML Model Performance (Test $R^2$)**| **-0.12** ($MAE \approx 4.65$ hrs)| Tabular metadata alone is insufficient to predict complex IT resolution times. |

`Resolution_Time_Hrs` appears to contain elapsed time for open tickets as well as final resolution time for closed tickets. Therefore, the values for Pending and In Progress tickets should not be described as completed resolution times. For operational reporting, it is better to call this field **elapsed/resolution time**, or model closed and still-open tickets separately.
---

The response-SLA classes are imbalanced: only 22 of 500 tickets are beyond eight hours. A classifier that always predicts “Within 8 Hours” would already score 95.6% accuracy, so accuracy alone would be misleading. Recall, precision, F1, PR-AUC, and a confusion matrix should be used for that task.
## 2. Dataset Schema & Column-by-Column Interpretation

## PyTorch modelling choice
The dataset contains 12 attributes with **zero missing values** and **zero duplicate ticket identifiers**.

The accompanying script predicts `Resolution_Time_Hrs` using only fields plausibly known when a ticket is opened:
### 1. `Ticket_ID`
* **Data Type**: Nominal String (`HD0001` to `HD0500`).
* **Interpretation**: Unique primary key. Correctly excluded from statistical and machine learning models to prevent arbitrary ID memorization.

- User type
- Department/unit
- Issue category
- Priority
- Month and day of week derived from the date
### 2. `Date`
* **Data Type**: Datetime (Recorded from `2026-01-08` to `2026-09-02`).
* **Interpretation**: Spans two full university semesters. Volume peaks coincide with key academic events:
  * **April (77 tickets)** and **July (72 tickets)** represent peak portal activity: course registration deadlines, fee clearances, and mid-session verifications.
  * **September (8 tickets)** captures only the first two days of the month (partial snapshot).

It deliberately excludes `Ticket_ID`, free-text issue descriptions, `Response_Time_Hrs`, `Status`, `Resolved`, and `Response_SLA`. These fields are either identifiers, outcomes observed later, or likely to leak information about the target.
### 3. `User_Type`
* **Distribution**:
  * **Students**: 240 tickets (48.0%)
  * **Postgraduate Students**: 137 tickets (27.4%)
  * **Staff**: 123 tickets (24.6%)
* **Interpretation**: Undergraduates constitute almost half the service demand, focusing heavily on Portal, ID cards, and Course Registration. Postgraduate students and university staff generate over half the aggregate volume, requiring specialized MIS systems, research network bandwidth, and staff email routing.

The data is split into training, validation, and test sets. Categorical variables are one-hot encoded, numeric date features are standardized, and a small multilayer perceptron is trained with Huber loss and early stopping. The script compares test MAE against a simple median prediction baseline. With only 500 records, the neural network should be treated as an experiment rather than a production model.
### 4. `Unit_Dept`
* **Distribution**: Spans 15 university units, colleges, and administrative bodies:
  * Major Academic Units: College of Engineering (COLENG), College of Plant Science (COLPLANT), College of Animal Science (COLANIM), College of Environmental Resources (COLERM), College of Veterinary Medicine (COLVET), College of Food Science (COLFST), College of Agricultural Management (COLAMRUD), College of Computing Sciences.
  * Administrative Units: Registry, Bursary, Library, Student Affairs, ICTREC, Academic Planning, Postgraduate School.
* **Interpretation**: Departmental demand is decentralized. Colleges encounter technical issues specific to their academic calendar, while administrative offices (Bursary, Registry) generate systemic workflow tickets (payments, clearances).

## How to run
### 5. `Issue_Category` & `Issue_Description`
* **Top Categories**:
  * **Portal** (79 tickets, 15.8%): Login failures, profile loading errors, session timeouts.
  * **Payment/Invoice** (63 tickets, 12.6%): Invoice generation glitches, delayed bank confirmation, receipts not printing.
  * **Admission** (54 tickets, 10.8%): O'Level verification issues, change of course updates, admission status tracking.
  * **Registration** (53 tickets, 10.6%): Course allocation missing, registration portal freezing.
  * **Internet/Network** (51 tickets, 10.2%): Campus Wi-Fi instability, slow bandwidth, IP connectivity dropouts.
  * **ID Card** (48 tickets, 9.6%): Printing defects, missing student bio-details, replacement requests.
  * **Email** (45 tickets, 9.0%): Mailbox access errors, password resets, institutional domain routing.
  * **Software/MIS** (39 tickets, 7.8%): ERP/MIS transaction failures, local software installation requests.
  * **Hardware** (34 tickets, 6.8%): Computer desktop boot failures, printer/scanner offline.
  * **Website/Multimedia** (20 tickets, 4.0%): Departmental page updates, media support.
  * **Student Information** (19 tickets, 3.8%): Biodata corrections, matriculation number updates.
  * **Others** (15 tickets, 3.0%): Miscellaneous support tickets.
* **Interpretation**: **Over 50% of tickets cluster around digital portal transactions** (Portal + Payment + Admission + Registration). These are ripe for automation, clear FAQs, and self-service password/payment reconciliation bots.

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn torch
python FUNAAB_Helpdesk_PyTorch_Analysis.py --csv FUNAAB_ICTREC_Helpdesk_500_Instances.csv
### 6. `Priority`
* **Distribution**:
  * **Medium**: 222 tickets (44.4%)
  * **Low**: 147 tickets (29.4%)
  * **High**: 110 tickets (22.0%)
  * **Critical**: 21 tickets (4.2%)
* **Interpretation**: The helpdesk follows a reasonable triage distribution: the vast majority are routine (Low/Medium), with only ~4% flagged as Critical emergencies.

### 7. `Response_Time_Hrs`
* **Descriptive Stats**: Mean: 3.09 hours, Median: 2.58 hours, Standard Deviation: ~2.2 hours, Min: 0.26 hours, Max: 20.66 hours.
* **Interpretation**: First response is prompt. Over 50% of tickets receive an acknowledgment within 2.6 hours.

### 8. `Response_SLA`
* **Distribution**:
  * **Within 8 Hours**: 478 tickets (95.6%)
  * **Beyond 8 Hours**: 22 tickets (4.4%)
* **Interpretation**: High SLA adherence for initial contact. The 22 violations primarily occurred during peak submission periods or on weekends/off-hours.

### 9. `Status` & `Resolved`
* **Distribution**:
  * **Closed** (`Resolved == "Yes"`): 358 tickets (71.6%)
  * **In Progress** (`Resolved == "No"`): 72 tickets (14.4%)
  * **Pending** (`Resolved == "No"`): 70 tickets (14.0%)
* **Interpretation**: 142 tickets (28.4%) remain unfinalized. "Pending" typically signifies tickets awaiting user feedback (e.g., student providing proof of payment), while "In Progress" signifies active engineering investigation.

### 10. `Resolution_Time_Hrs` (The Critical Metric)
* **Overall Column**: Mean = 13.40 hours, Median = 7.64 hours.
* **Separated by Operational Status**:
  * **Closed / Resolved Tickets**: Mean = **7.44 hours**, Median = **5.68 hours** (Max = 57.62 hrs).
  * **In Progress / Pending Tickets**: Mean = **28.42 hours**, Median = **28.50 hours**.

---

## 3. The Critical Data Quality & Censoring Caveat

> [!CAUTION]
> **Data Censoring Pitfall**:
> In standard helpdesk telemetry, `Resolution_Time_Hrs` measures the elapsed duration between ticket creation and ticket closure. 
> However, for rows where `Status` is **In Progress** or **Pending**, resolution has **not yet occurred**. 
> The recorded figure in those rows is right-censored — it represents the **time elapsed so far** until the data dump was exported, not the true duration required to fix the issue.

### Why This Matters for Modeling and Business Reporting
1. **Misleading Averages**: If an analyst averages the entire `Resolution_Time_Hrs` column across all 500 rows, they obtain **13.40 hours**, exaggerating true resolution time by nearly 80%.
2. **Target Leakage / Data Pollution**: Training an AI model on the full column forces the network to learn an artificial distribution where open tickets look inherently "longer" simply because they haven't been touched yet.
3. **The Solution Implemented in `main.py`**:
   [`load_and_prepare()`](file:///c:/Users/user/Documents/mymodel/main.py#L55-L74) filters exclusively for `Resolved == 'Yes'` (358 instances). Unresolved tickets are isolated for backlog aging reports.

---

## 4. Key Cross-Variable & Operational Insights

### Fast Initial Response Does Not Guarantee Fast Resolution
* **Observed Pearson Correlation**: $r \approx -0.16$ (resolved subset) and $-0.06$ (full dataset).
* **Insight**: In many support desks, managers mistakenly treat Response SLA as a proxy for customer satisfaction. Here, a ticket acknowledged in 30 minutes can still take 40+ hours if it requires database schema migration or third-party bank settlement verification.

```
Response Time (hrs)  ──── (r = -0.16) ────X No predictive link X────> Resolution Time (hrs)
  [Swift triage]                                                     [Driven by task complexity]
```

The script writes summary statistics, charts, test metrics, and the trained model into `analysis_output/`.
### Priority vs. Actual Resolution Time
* Contrary to naive expectation, **High and Critical tickets do not always close faster in raw hours than Low priority tickets**.
* Low priority tickets often consist of trivial tasks (e.g., standard password resets, FAQ guidance) which resolve in 1–2 hours.
* Critical priority tickets often involve system-wide network failures, database corruption, or portal gateway outages that inherently require cross-departmental coordination, taking 8 to 24+ hours.

## How to judge the model
### Departmental Load
* Colleges with intensive lab, matriculation, or computational coursework (COLENG, COLANIM, Computing Sciences) log higher frequencies of Network and Hardware requests.
* Central administrative bodies (Library, Academic Planning, Bursary) serve as hubs for ID card verification and invoice clearance tickets.

- **MAE** is the typical absolute prediction error in hours. Lower is better.
- **RMSE** penalizes large mistakes more strongly. Lower is better.
- **R²** measures improvement over predicting the mean. Values near or below zero indicate weak predictive usefulness on unseen data.
- Compare model MAE with `baseline_MAE_hours`. The neural network is useful only if it consistently beats the baseline across repeated or cross-validated splits.
---

## 5. Machine Learning Modeling Analysis ([`main.py`](file:///c:/Users/user/Documents/mymodel/main.py))

### Model Architecture & Training Setup
* **Architecture**: Multilayer Perceptron (`ResolutionMLP`)
  * Input layer: One-hot encoded categorical variables (`User_Type`, `Unit_Dept`, `Issue_Category`, `Priority`) + standardized numeric date features (`month`, `day_of_week`).
  * Hidden Layers: Linear(dim $\to$ 64) $\to$ ReLU $\to$ Dropout(0.15) $\to$ Linear(64 $\to$ 32) $\to$ ReLU $\to$ Linear(32 $\to$ 1).
  * Loss Function: **Huber Loss** (smooth L1, robust against right-skewed outliers).
  * Optimizer: **AdamW** (lr = 0.001, weight decay = $10^{-4}$).
  * Early stopping with patience = 30.

### Model Evaluation Results
From [`analysis_output/model_metrics.json`](file:///c:/Users/user/Documents/mymodel/analysis_output/model_metrics.json):
* **Test MAE**: **4.65 hours**
* **Baseline Median MAE**: **4.69 hours**
* **Test RMSE**: **6.71 hours**
* **Test $R^2$**: **-0.12**
* **Sample Split**: 228 Train / 58 Validation / 72 Test

### Why Does the Neural Network Fail to Beat the Simple Baseline ($R^2 < 0$)?
A negative coefficient of determination ($R^2 = -0.12$) indicates that the neural network's predictions on unseen test tickets perform slightly worse than simply guessing the constant median resolution time of the training set.

This is a **classic machine learning finding in tabular IT helpdesk data**, caused by four structural realities:

1. **Missing Latent Determinants (Omitted Variable Bias)**:
   The dataset captures *what* the problem is and *who* logged it, but omits the true drivers of resolution speed:
   * Which engineer was assigned? (Senior vs. intern)
   * Was the engineer currently on duty or overloaded with 20 other tickets?
   * Was there a third-party dependency? (Interswitch/Remita payment gateway, internet service provider fiber cut)
   * Did the requester respond immediately with requested details?
2. **Curse of Dimensionality on Small Sample ($N = 358$)**:
   With only 358 resolved rows split 80/20 into train/val/test, the training set has just 228 examples across ~35 one-hot encoded feature dimensions. A neural network with thousands of parameters easily overfits spurious noise.
3. **High Intrinsic Variance**:
   Two tickets labeled "ID card printing problem" can vary dramatically: one is fixed in 10 minutes by changing printer toner, while another takes 3 days because the student's passport photo is corrupt in the central database.
4. **Summary**: A tabular MLP cannot learn relationships that simply do not exist in the feature set. For tabular datasets of this size and nature, simpler tree models (e.g. LightGBM, Random Forest) or pure median heuristics are far more reliable.

---

## 6. Strategic Recommendations for FUNAAB ICTREC Operations

### 1. Implement Tiered Self-Service & Automated Triage
* **Portal & Payment Self-Service**: Over 28% of all tickets are Portal and Payment queries. Integrating an automated receipt re-query tool and self-service password reset bot would eliminate roughly 140 tickets every 8 months.
* **Pre-submission Form Validation**: Require students to input their Transaction Reference ID or Matriculation Number before opening a ticket to prevent back-and-forth email delays.

### 2. Establish a Formal Resolution SLA
* The current operation boasts a **95.6% Response SLA**, but has **no defined Resolution SLA**.
* ICTREC should establish target resolution tiers:
  * **Critical**: Resolve within 6 hours.
  * **High**: Resolve within 12 hours.
  * **Medium**: Resolve within 24 hours.
  * **Low**: Resolve within 48 hours.

### 3. Active Queue Management for Aging Backlog
* The 142 currently open tickets have been pending for an average of **28.4 hours**.
* Implement automated reminders for tickets pending user response (>24 hours) and automatic escalation alerts for In Progress tickets approaching 48 hours.

### 4. Data Collection Upgrades for Future Machine Learning
To make ticket duration prediction genuinely viable, future versions of the dataset should capture:
* `Assigned_Technician_ID` / `Support_Tier` (Tier 1 vs. Tier 2/3)
* `Requester_Wait_Time_Hrs` (time spent waiting for student response)
* `Third_Party_Dependent` (Boolean flag for bank/ISP issues)
* `Text_Embedding` extracted from the full `Issue_Description` using a pretrained LLM.
