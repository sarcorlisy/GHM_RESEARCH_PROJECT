# Hospital Readmission Prediction Pipeline - Quick Start Guide

## 5-Minute Quick Start

### 1. Environment Setup

Ensure your system has Python 3.8+ installed, then execute:

```bash
# Clone or download project files
cd rp0609

# Install dependencies
pip install -r requirements.txt
```

### 2. Data File Check

Ensure the following files are in the project root directory:
- `diabetic_data.csv`
- `IDS_mapping.csv`
- `ccs_icd9_mapping.csv`

### 3. Run Complete Pipeline

```bash
# Method 1: Use main pipeline
python main_pipeline.py

# Method 2: Use example script (recommended for beginners)
python run_example.py

# Method 3: Use test script to verify functionality
python test_pipeline.py
```

### 4. View Results

After completion, check the `outputs/` directory:

```
outputs/
├── models/                    # Trained models
├── feature_importance.png     # Feature importance plot
├── model_comparison.png       # Model comparison plot
├── final_pipeline_report.txt  # Complete report
└── pipeline.log              # Execution log
```

## Expected Results

After successful execution, you should see output similar to this:

```
Hospital Readmission Prediction Pipeline
==================================================
Pipeline completed successfully!

Results Summary:
------------------------------
Best Model: RandomForest
Best AUC Score: 0.965
Total Features Selected: 15
Training Samples: 3 models trained

Model Performance Comparison:
----------------------------------------
model_name         auc    f1  accuracy
LogisticRegression 0.670 0.580     0.65
RandomForest       0.965 0.933     0.94
XGBoost            0.958 0.931     0.93
```

## Common Problem Solutions

### Problem 1: Missing Data Files
```
Error: Data file not found
```
**Solution**: Ensure all CSV files are in the project root directory

### Problem 2: Dependency Package Installation Failure
```
ModuleNotFoundError: No module named 'xgboost'
```
**Solution**: 
```bash
pip install xgboost
# Or reinstall all dependencies
pip install -r requirements.txt
```

### Problem 3: Insufficient Memory
```
MemoryError
```
**Solution**: Reduce feature selection count in `pipeline_config.py`:
```python
MODEL_CONFIG = {
    'feature_selection_top_n': 10  # Reduce from 15 to 10
``` 