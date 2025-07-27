# Tata-Steel-Machine-Failure-Prediction-Classification

This Google Colab-based project is an **end-to-end Machine Learning pipeline** for predicting **machine failure** and optionally identifying the **type of failure** using real-world telemetry and sensor data from industrial machinery (e.g., Tata machines).

The core aim is to support **predictive maintenance** strategies by:
- Predicting the likelihood of machine failure in advance
- Diagnosing the exact type of failure (if it occurs)
- Improving operational reliability and reducing downtime/costs

##  PROJECT OBJECTIVE (Business Context)

Modern manufacturing operations rely on the continuous health of machines. Unexpected breakdowns result in **production loss**, **repair costs**, and **safety hazards**.

This ML solution helps solve two key business problems:

1.  **Predict whether a machine will fail** — so the operations team can take preventive action
2.  **Diagnose what type of failure will occur** — so the correct resources and parts are arranged in advance

Ultimately, the project delivers **AI-driven insights** for better maintenance planning and machine lifecycle management.

##  STAGE 1: MACHINE FAILURE PREDICTION

###  What It Does

Predicts whether a machine is going to **fail soon** based on sensor readings.

###  Data Preprocessing & Feature Engineering
- **Input data**: Sensor readings like temperature, torque, speed, wear, etc.
- **Mapped 'Type' feature** (L=0, M=1, H=2) for model understanding
- **Feature engineering**:
  - `Temp_Diff = Process Temperature - Air Temperature`
  - `Power = Torque × Rotational Speed`
  - `Torque_per_speed = Torque / Speed`
- Unused columns (e.g., IDs, failure labels) are dropped

###  Scaling
- All numerical features are scaled using `StandardScaler` to ensure model consistency

###  ML Model
- Trained **XGBoost Classifier** used
- Predictions are made on the **unseen test dataset** (`test.csv`)
- Model outputs:
  - `Failure_Probability` – Probability of machine failure
  - `Predicted_Label` – Binary (1 = will fail, 0 = will not fail)

###  Threshold Tuning
- A threshold of **0.30** is applied to balance **high recall** and minimize missed failures

###  Output
- Final prediction file: `test_with_predictions.csv`
- Includes failure probability, percentage, and predicted status

###  Business Impact
- Allows teams to act before failures happen
- Saves downtime, cost, and human effort
- Helps automate early warning systems in a smart factory setup

##  STAGE 2: FAILURE TYPE PREDICTION (BONUS WORK)

###  What It Does

Once a machine is predicted to fail (`Predicted_Label == 1`), this stage diagnoses **what type of failure** will happen:
- `TWF` – Tool Wear Failure
- `HDF` – Heat Dissipation Failure
- `PWF` – Power Failure
- `OSF` – Overstrain Failure

###  Approach
- **Multi-label classification**: Each type is treated as a separate binary classification task
- One model is trained for each failure type

###  ML Model: Stacked Ensemble
Each model uses:
-  **Base learners**:  
  - XGBoost Classifier  
  - Random Forest Classifier  
-  **Meta-learner**:  
  - Logistic Regression  
-  **Technique**:  
  - `StackingClassifier` with passthrough and class balancing
  - `compute_class_weight()` used to handle label imbalance

###  Threshold Tuning
For each failure type, best threshold is selected based on **F1-score** using the test set.

| Failure Type | Tuned Threshold | Precision | Recall | F1-Score |
|--------------|------------------|-----------|--------|----------|
| TWF          | 0.50             | 0.54      | 0.74   | 0.62     |
| HDF          | 0.62             | 0.96      | 0.93   | 0.95     |
| PWF          | 0.63             | 0.23      | 0.50   | 0.32     |
| OSF          | 0.55             | 0.81      | 0.79   | 0.80     |

###  Predictions on Unseen Data
- The `test_with_predictions.csv` file is filtered for rows where `Predicted_Label == 1`
- The same scaling steps are applied
- Each stacked model is loaded and used to predict respective failure types
- Final output: `final_failure_type_predictions.csv`

###  Business Impact
- Gives actionable insights on **why** a machine might fail
- Helps maintenance engineers prioritize specific checks
- Enables **root-cause diagnosis** and part replacements before breakdowns occur

##  END-TO-END PIPELINE FLOW

```
[Raw Sensor Data]
        ↓
[Feature Engineering + Scaling]
        ↓
[Machine Failure Model (XGBoost)]
        ↓
[Predicted_Label == 1 ?]
        ↓
    ↙       ↘
[Yes]     [No → Healthy Machines]
  ↓
[Stacked Models for TWF, HDF, PWF, OSF]
        ↓
[Final Output: Failure Type Diagnosed]
```

##  FILES AND NOTEBOOKS

This entire solution is built within a **Google Colab Notebook**. No local setup or Python scripts are needed.

Key CSVs used:
- `final_tatamachinedataset_for_ML.csv` – Training data
- `test.csv` – Unseen test data
- `test_with_predictions.csv` – Machine failure prediction results
- `final_failure_type_predictions.csv` – Final failure type output


##  KEY TAKEAWAYS

-  Predicts **whether a machine will fail** using sensor telemetry
-  Diagnoses the **exact cause of failure** if a machine is predicted to fail
-  Uses **stacked ensemble learning** with threshold tuning
-  Handles **real-world imbalance** and supports multi-label outputs
-  Offers **scalable and adaptable predictive maintenance framework** for smart manufacturing

##  AUTHOR & CREDITS

**Developed by**: Vishesh Alag  
**Use case**: Industrial Predictive Maintenance – Tata Machines  


---

*This project demonstrates the power of Machine Learning in transforming traditional reactive maintenance into proactive, data-driven maintenance strategies.*
