# FUNAAB ICTREC Helpdesk Data Interpretation

## What the data contains

The file contains 500 tickets recorded from 8 January to 2 September 2026. It has 12 columns and no missing values. Each ticket records the requester type, department, issue, priority, response and resolution times, status, resolution flag, and response SLA result.

## Main observations

- 358 tickets were resolved, giving a resolution rate of **71.6%**.
- 478 tickets received a response within eight hours, giving a response-SLA compliance rate of **95.6%**.
- Mean response time was **3.09 hours**; median response time was **2.58 hours**.
- Mean resolution time was **13.40 hours**; median resolution time was **7.64 hours**. The mean being much larger than the median indicates a right-skewed distribution with some long-running tickets.
- The most frequent category was **Portal** with 79 tickets, followed by **Payment/Invoice** with 63 and **Admission** with 54.
- Priority distribution was Medium 222, Low 147, High 110, and Critical 21.
- Students submitted 240 tickets, postgraduate students 137, and staff 123.
- Closed tickets had a median resolution time of **5.68 hours**. In-progress and pending tickets had much higher median recorded durations of **28.03** and **29.77 hours** respectively.
- Response time and resolution time had a Pearson correlation of about **-0.06**, so this sample does not show a useful linear relationship between those two fields.

## Important interpretation cautions

`Resolution_Time_Hrs` appears to contain elapsed time for open tickets as well as final resolution time for closed tickets. Therefore, the values for Pending and In Progress tickets should not be described as completed resolution times. For operational reporting, it is better to call this field **elapsed/resolution time**, or model closed and still-open tickets separately.

The response-SLA classes are imbalanced: only 22 of 500 tickets are beyond eight hours. A classifier that always predicts “Within 8 Hours” would already score 95.6% accuracy, so accuracy alone would be misleading. Recall, precision, F1, PR-AUC, and a confusion matrix should be used for that task.

## PyTorch modelling choice

The accompanying script predicts `Resolution_Time_Hrs` using only fields plausibly known when a ticket is opened:

- User type
- Department/unit
- Issue category
- Priority
- Month and day of week derived from the date

It deliberately excludes `Ticket_ID`, free-text issue descriptions, `Response_Time_Hrs`, `Status`, `Resolved`, and `Response_SLA`. These fields are either identifiers, outcomes observed later, or likely to leak information about the target.

The data is split into training, validation, and test sets. Categorical variables are one-hot encoded, numeric date features are standardized, and a small multilayer perceptron is trained with Huber loss and early stopping. The script compares test MAE against a simple median prediction baseline. With only 500 records, the neural network should be treated as an experiment rather than a production model.

## How to run

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn torch
python FUNAAB_Helpdesk_PyTorch_Analysis.py --csv FUNAAB_ICTREC_Helpdesk_500_Instances.csv
```

The script writes summary statistics, charts, test metrics, and the trained model into `analysis_output/`.

## How to judge the model

- **MAE** is the typical absolute prediction error in hours. Lower is better.
- **RMSE** penalizes large mistakes more strongly. Lower is better.
- **R²** measures improvement over predicting the mean. Values near or below zero indicate weak predictive usefulness on unseen data.
- Compare model MAE with `baseline_MAE_hours`. The neural network is useful only if it consistently beats the baseline across repeated or cross-validated splits.

