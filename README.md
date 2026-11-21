# Customer Churn Prediction

A small end-to-end machine learning project predicting customer churn for a telecommunications provider using the **IBM Telco Customer Churn** dataset. It compares three classical scikit-learn baselines with a simple neural network built in PyTorch.

The project focuses on reproducible preprocessing, a leakage-aware train/validation/test workflow, threshold selection, and the business trade-off between missing customers who churn and contacting customers who would stay.

## Highlights

- **7,043 customers** and 21 columns in the original dataset; **26.54%** of customers churned.
- Stratified **60% train / 20% validation / 20% test** split (`random_state=42`).
- Four models: Dummy Classifier, Logistic Regression, Random Forest, and PyTorch MLP.
- A scikit-learn `Pipeline` and `ColumnTransformer` for categorical encoding and numerical scaling.
- PyTorch MLP trained with `BCEWithLogitsLoss`, Adam, and early stopping based on validation loss.
- Model and decision-threshold selection performed on validation data, followed by a final evaluation on held-out test data.

## Dataset and exploratory analysis

The project uses the **IBM Telco Customer Churn** dataset, stored locally at:

`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

The dataset contains customer demographics, account information, subscribed services, contract details, and a binary `Churn` label.

During EDA:

- No duplicate rows or duplicate `customerID` values were found.
- `TotalCharges` was stored as text; 11 blank values occurred for customers with `tenure = 0`. These were converted to numeric values and filled with `0`.
- The dataset contains **5,174 non-churned customers (73.46%)** and **1,869 churned customers (26.54%)**.
- Observed churn rates by contract type were **42.71%** for month-to-month, **11.27%** for one-year, and **2.83%** for two-year contracts.
- Churned customers tended to have shorter tenure and higher monthly charges.

These are descriptive associations, **not evidence of causality**. Exploratory analysis is available in [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb).

## Project structure

```text
customer-churn-prediction/
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── preprocess.py
│   ├── train_sklearn.py
│   ├── train_pytorch.py
│   ├── evaluate.py
│   ├── final_evaluation_logistic_regression.py
│   └── final_evaluation_neural_network.py
├── tests/
│   └── test_preprocessing.py
├── pyproject.toml
└── README.md
```

The tree focuses on the files used by the project; local environment and Python cache directories are omitted.

## Setup

**Requirements:** Python 3.13 and the dependencies declared in `pyproject.toml`.

From the repository root:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the project and its dependencies:

```bash
python -m pip install -e .
```

If `pyproject.toml` defines a `dev` optional extra, install the development dependencies (including `pytest`) with:

```bash
python -m pip install -e ".[dev]"
```

The raw dataset CSV must be present at the path shown above. Run all commands below from the **repository root**, since the scripts use repository-relative paths.

## Methodology

### Data preparation and leakage prevention

1. Convert `TotalCharges` to a numerical feature and fill the 11 blank values with zero (all correspond to `tenure = 0`).
2. Map `Churn` from `No`/`Yes` to `0`/`1` and remove `customerID` from model features.
3. Create a single stratified split: **4,225 train / 1,409 validation / 1,409 test** rows.
4. Fit numerical `StandardScaler` and categorical `OneHotEncoder(handle_unknown="ignore")` transformations on **training data only**.
5. Apply the already fitted transformations to validation and test data; the 19 input features expand to **45 encoded features**.

Classical models use scikit-learn pipelines. The PyTorch MLP uses the same training-only preprocessing and converts transformed features and labels into `float32` tensors. The validation set is used for model comparison, the MLP's early stopping, and threshold selection. The test set is reserved for final reporting.

### Models

| Model | Configuration |
|---|---|
| Dummy Classifier | `strategy="most_frequent"` |
| Logistic Regression | `max_iter=1000`, `random_state=42` |
| Random Forest | `n_estimators=100`, `random_state=42` |
| PyTorch MLP | `45 → 32 → ReLU → 1`, Adam (`lr=0.001`), `BCEWithLogitsLoss`, up to 500 full-batch epochs, early stopping (`patience=20`) |

The MLP outputs a **logit**. Sigmoid is applied to obtain a probability for evaluation; `BCEWithLogitsLoss` takes logits directly during training. The best MLP weights are restored based on validation loss.

### Evaluation metrics

- **Accuracy:** fraction of correct predictions.
- **Precision:** fraction of predicted churners who actually churned.
- **Recall:** fraction of actual churners detected by the model.
- **F1-score:** harmonic mean of precision and recall.
- **ROC-AUC:** ranking performance across decision thresholds.
- **Average Precision (AP):** summary of the precision–recall curve, calculated with scikit-learn's `average_precision_score` (not trapezoidal PR-AUC).

Precision, recall, F1, and accuracy depend on the decision threshold. ROC-AUC and Average Precision are calculated from predicted probabilities.

## Results

### Validation: model comparison

The following metrics were computed on the **same validation set** with a **0.5 classification threshold** for the threshold-dependent metrics.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|
| Dummy Classifier | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 | 0.265 |
| Logistic Regression | **0.823** | **0.697** | **0.591** | **0.640** | **0.860** | **0.678** |
| Random Forest | 0.787 | 0.634 | 0.468 | 0.538 | 0.831 | 0.623 |
| PyTorch MLP | 0.817 | 0.684 | 0.578 | 0.626 | 0.858 | 0.672 |

### Validation: decision threshold

A lower threshold prioritizes **recall**, identifying more customers who may churn at the expense of more false positives.

| Model | Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.3 | 0.534 | **0.791** | 0.638 |
| Logistic Regression | 0.5 | 0.697 | 0.591 | 0.640 |
| Logistic Regression | 0.7 | 0.793 | 0.184 | 0.299 |
| PyTorch MLP | 0.3 | 0.541 | 0.781 | 0.639 |
| PyTorch MLP | 0.5 | 0.684 | 0.578 | 0.626 |
| PyTorch MLP | 0.7 | 0.777 | 0.214 | 0.335 |

**Selected model: Logistic Regression, threshold = 0.3.** This selection was made using validation results, prioritizing recall, while also considering model simplicity and the very small differences in ranking metrics. There is no claim that its performance is statistically significantly better than that of the MLP.

### Final held-out test results

Both models below were evaluated using the **preselected threshold of 0.3**. The MLP test results are reported as an additional comparison; they were **not used to revise the selected model**.

| Metric | Logistic Regression (selected) | PyTorch MLP (comparison) |
|---|---:|---:|
| Accuracy | 0.755 | **0.764** |
| Precision | 0.527 | **0.539** |
| Recall | **0.770** | **0.770** |
| F1-score | 0.625 | **0.634** |
| ROC-AUC | 0.833 | **0.834** |
| Average Precision | **0.622** | 0.620 |

The selected Logistic Regression model achieved this test confusion matrix (`rows = actual`, `columns = predicted`; labels `0 = stay`, `1 = churn`):

| Actual / Predicted | Stay (0) | Churn (1) |
|---|---:|---:|
| Stay (0) | **776** (TN) | **259** (FP) |
| Churn (1) | **86** (FN) | **288** (TP) |

This means the selected model identified **288 of 374** actual churners (77.0% recall) while flagging **547 of 1,409** test customers for potential follow-up. Among those 547 flagged customers, 288 actually churned (52.7% precision).

## Business interpretation and limitations

At the selected 0.3 threshold, the model prioritizes identifying customers at risk of churn. The trade-off is that **259 customers who stayed** were incorrectly flagged in the test set, which could generate unnecessary outreach costs. At the same time, **86 actual churners** were missed.

This is a **churn-risk prediction model, not a causal model or a retention-uplift model**. It does not estimate whether a discount or contact would prevent churn. An operational campaign would require assumptions or evidence about intervention costs, customer value, and treatment effectiveness.

Other limitations:

- Results come from a single dataset and one stratified split; performance may differ across samples, providers, or time periods.
- Churn is imbalanced (26.54% positive cases), so accuracy alone is insufficient.
- No extensive hyperparameter tuning, calibration study, cross-validation, or deployment was performed.
- Small differences between the models should not be interpreted as statistically significant without further analysis.
- The final test set is for reporting, not for selecting another architecture or decision threshold.

## Reproduce the experiments

Run from the project root:

```bash
# Compare classical models on validation and inspect logistic thresholds
python src/evaluate.py

# Train and evaluate the PyTorch MLP on validation
python src/train_pytorch.py

# Final held-out test evaluation for the selected model
python src/final_evaluation_logistic_regression.py

# Additional held-out test comparison for the MLP
python src/final_evaluation_neural_network.py
```

The final-evaluation scripts reproduce the relevant training steps before evaluating the models; no pre-trained model files are required. The neural-network initialization uses a fixed PyTorch seed for repeatability within the same environment.

### Run tests

The project includes small `pytest` checks for data cleaning, immutability of the original DataFrame, and disjoint train/validation/test splits.

```bash
# Linux / macOS
PYTHONPATH=src python -m pytest -v
```

```powershell
# Windows PowerShell
$env:PYTHONPATH="src"
python -m pytest -v
```

## Tools

Python 3.13 · pandas · NumPy · scikit-learn · PyTorch · Matplotlib · Seaborn · pytest
