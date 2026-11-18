# Predicting Late Deliveries Before Dispatch

Classification project: predict whether a shipment will be Late or On-Time before it is dispatched.

## Problem
A logistics company wants to catch late deliveries early. The model uses shipment details, route logs and carrier performance logs.

## Approach
- Merged 3 tables on shipment_id and summarised the logs to one row per shipment
- Cleaned data: filled missing origin hubs, fixed negative route delays, capped outliers at the 99th percentile
- Features: delay_mean, delay_max, delay_sum, log_count, carrier_mean, carrier_min, plus one-hot encoded shipment type and origin hub
- Scikit-learn pipelines (median imputation, min-max scaling, model)
- Compared Logistic Regression, Decision Tree, Random Forest and Gradient Boosting with 5-fold cross-validation, then tuned them with GridSearchCV
- Chose a decision threshold of 0.42 so the model catches at least 75% of late shipments

## Results (20% test set)
- Selected model: Logistic Regression
- ROC-AUC: 0.788
- Late class: recall 0.83, precision 0.40
- Most important features: delay_mean, carrier_mean, delay_sum, carrier_min

The model catches most late shipments, at the cost of more false alarms.

## Tools
Python, pandas, scikit-learn, matplotlib

## Notes
Built as a training project with AI assistance for drafting code. I reviewed and understand each step.
Data files are not included. To run the notebook, place the 5 CSV files in the same folder and change DATA_DIR in the first cell.
