# Hospital Readmissions Reduction Program: Predictive Modeling and Utility Analysis

This repository contains a Jupyter Notebook (`sripfinal.ipynb`) that demonstrates an end-to-end machine learning pipeline for predicting excess hospital readmissions. The project leverages data from the Hospital Readmissions Reduction Program to build, evaluate, and interpret a predictive model, focusing on both performance metrics and real-world utility in terms of cost savings.

The analysis includes:
* **Data Preprocessing**: Handling missing values and encoding categorical features.
* **Hyperparameter Optimization**: Using Optuna to fine-tune the XGBoost model.
* **Model Evaluation**: Comprehensive assessment using various metrics.
* **Model Explainability**: Interpreting predictions with SHAP.
* **Calibration Analysis**: Assessing how well predicted probabilities align with actual outcomes.
* **Fairness Analysis**: Examining model performance across different states.
* **Utility Analysis**: Quantifying the financial impact of the model's predictions.
* **Baseline Comparison**: Benchmarking the XGBoost model against other common classifiers.

## Dataset

The project utilizes data from the `Hospital_Readmissions_Reduction_Program_Hospital.csv` file. This dataset contains information related to hospital readmission rates, including excess readmission ratios, various hospital characteristics, and state information.

**Key Target Variable**: `readmit_flag`
* Derived from `Excess Readmission Ratio > 1.0`, indicating whether a hospital has an "excessive" readmission rate (1) or not (0).

## Methodology and Key Features

1.  **Data Loading and Cleaning**:
    * Loads the hospital data.
    * Drops "leaky" columns that might directly or indirectly reveal the target variable, preventing data leakage (e.g., `Predicted Readmission Rate`, `Expected Readmission Rate`, `Number of Readmissions`).
    * Removes rows with missing `Excess Readmission Ratio`.
    * Converts `Excess Readmission Ratio` to a float.

2.  **Feature Engineering & Preprocessing Pipeline**:
    * Separates numerical and categorical features.
    * **Numerical Features**: Missing values imputed with the mean, then scaled using `StandardScaler`.
    * **Categorical Features**: Missing values imputed with the most frequent value, then one-hot encoded using `OneHotEncoder`.
    * A `ColumnTransformer` integrates these preprocessing steps into a single, robust pipeline.

3.  **Data Splitting**:
    * The dataset is split into training and testing sets (70% train, 30% test) using `train_test_split` with `stratify=y` to maintain the class distribution in both sets.

4.  **Model Training and Hyperparameter Optimization**:
    * **XGBoost Classifier** is chosen as the primary predictive model.
    * **Optuna** is used for automated hyperparameter tuning to find the best `n_estimators`, `max_depth`, `learning_rate`, `subsample`, and `colsample_bytree` parameters.
    * Cross-validation (`cv=3`) with `roc_auc` scoring is used during optimization to ensure robust parameter selection.

5.  **Model Evaluation**:
    * After training the final XGBoost model with optimized parameters, its performance is evaluated on the test set.
    * Metrics reported include:
        * **AUC (Area Under the Receiver Operating Characteristic Curve)**: A primary metric for binary classifiers, indicating the model's ability to distinguish between classes.
        * **Accuracy**: Overall correctness of predictions.
        * **Brier Score Loss**: Measures the accuracy of probabilistic predictions.
        * **Confusion Matrix**: Provides a breakdown of true positives, true negatives, false positives, and false negatives.
        * **Classification Report**: Shows precision, recall, and F1-score for each class.
    * **Bootstrap AUC Confidence Interval**: Calculates a 95% confidence interval for the AUC using bootstrapping (1000 resamples) on the test set for a more reliable estimate.

6.  **Model Explainability (SHAP)**:
    * **SHAP (SHapley Additive exPlanations)** values are computed to explain individual predictions and highlight the most important features driving the model's decisions.
    * `shap.plots.beeswarm` and `shap.plots.bar` visualizations are used to show feature importance.

7.  **Calibration Curve**:
    * A calibration curve is plotted to visualize how well the predicted probabilities from the model align with the true probabilities of the event. A well-calibrated model's predictions should fall close to the diagonal line.

8.  **Subgroup Fairness Analysis**:
    * The model's AUC performance is assessed for different `State` subgroups within the test data.
    * The "Top 10 States by AUC" are printed, highlighting potential disparities or varying model effectiveness across geographical regions.

9.  **Baseline Model Comparison**:
    * Three additional baseline models are trained and evaluated for comparison:
        * **Logistic Regression**
        * **Random Forest Classifier**
        * **Neural Network (MLPClassifier)**
    * ROC curves for these models are plotted together to visually compare their discriminative power.

10. **Utility Analysis**:
    * A crucial step to understand the practical value of the model.
    * Defines `cost_per_readmission` (`₹5000`) and `cost_per_intervention` (`₹1000`).
    * Calculates:
        * **True Positive Savings**: Savings from correctly identifying and preventing readmissions.
        * **False Positive Costs**: Costs incurred from intervening on patients predicted to be high-risk but who would not have been readmitted.
        * **Net Utility**: The overall financial benefit (Savings - Costs).
    * Visualizes these values as a stacked bar chart for clear interpretation.

## Results Summary (Example Output based on notebook)

* **XGBoost Model Performance**:
    * AUC: ~0.685
    * Accuracy: ~0.624
    * Brier Score: ~0.222
    * 95% AUC CI: [0.669, 0.701]
* **Top 10 States by AUC**: Shows variability in model performance across states, e.g., SD (0.923), MT (0.892), ID (0.853), etc.
* **Baseline Models AUCs**:
    * Logistic Regression AUC: ~0.662
    * Random Forest AUC: ~0.673
    * Neural Network AUC: ~0.638
* **Utility Analysis**:
    * True Positive Savings: ₹4,940,000
    * False Positive Costs: ₹593,000
    * **Net Utility: ₹4,347,000** (This indicates a significant potential financial benefit from using the model for interventions).

## Dependencies

To run this notebook, you will need the following Python libraries. You can install them using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap joblib optuna xgboost
