# Hospital Readmission Prediction Pipeline

A comprehensive hospital readmission prediction data science pipeline for predicting the risk of patients being readmitted within 30 days after discharge.

## Project Overview

This project refactors the original Jupyter notebook into a modular, reusable data science pipeline with the following main features:

- **Data Loading and Merging**: Automatic loading and merging of multiple data sources
- **Data Preprocessing**: Feature engineering, data cleaning, encoding, and standardization
- **Exploratory Data Analysis (EDA)**: Data distribution, correlation analysis, medical interpretation
- **Feature Selection**: Multiple feature selection methods (L1 regularization, mutual information, tree model importance)
- **Model Training**: Multiple machine learning models (Logistic Regression, Random Forest, XGBoost)
- **Model Evaluation**: Cross-validation, test set evaluation, performance comparison
- **Result Visualization**: Feature importance plots, model comparison plots
- **Report Generation**: Automatic generation of detailed training and evaluation reports
- **Sensitivity Analysis**: Model performance analysis for different patient subgroups

## Project Structure

```
rp0609/
├── pipeline_config.py          # Configuration file
├── data_loader.py              # Data loading module
├── data_preprocessor.py        # Data preprocessing module
├── feature_selector.py         # Feature selection module
├── model_trainer.py            # Model training module
├── eda_analyzer.py             # Exploratory data analysis module
├── main_pipeline.py            # Main pipeline file
├── Demo0720 main.ipynb         # Main execution file (complete analysis workflow)
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
├── outputs/                    # Output directory
│   ├── models/                 # Saved models
│   ├── *.png                   # Visualization charts
│   ├── *.txt                   # Report files
│   ├── *.csv                   # Processed data
│   └── *.xlsx                  # Detailed result tables
└── data/                       # Data files
    ├── diabetic_data.csv       # Main dataset
    ├── IDS_mapping.csv         # ID mapping data
    └── ccs_icd9_mapping.csv    # ICD-9 mapping data
```

## Installation and Setup

1. **Clone the project**
```bash
git clone <repository-url>
cd rp0609
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare data files**
Ensure the following data files are in the project root directory:
- `diabetic_data.csv`
- `IDS_mapping.csv`
- `ccs_icd9_mapping.csv`

## Usage

### 1. Main Execution Method (Recommended)

Open and run `Demo0720 main.ipynb`, which is the main execution file containing the complete analysis workflow:

- **Data Loading and Preview**: Raw data loading and quick preview
- **Data Filtering**: Keep first admission records, remove patients who cannot be readmitted
- **Data Preprocessing**: Feature engineering, missing value handling, encoding, and standardization
- **Exploratory Data Analysis**: Data distribution, correlation analysis, medical interpretation
- **Feature Selection**: Multiple Top N value feature selection and result display
- **Model Training**: GridSearchCV hyperparameter tuning
- **Result Saving**: Automatic saving to Excel files

### 2. Run Complete Pipeline

```bash
python main_pipeline.py
```

This will execute the complete data science workflow:
- Data loading and merging
- Feature engineering and preprocessing
- Feature selection
- Model training and evaluation
- Generate reports and visualizations

### 3. Sensitivity Analysis

Sensitivity analysis is performed using separate notebooks, with each subgroup analysis completed in an independent notebook:

- `Demo0713 sensitivity comorbidity1.ipynb`: Comorbidity subgroup 1 analysis
- `Demo0713 sensitivity comorbidity2.ipynb`: Comorbidity subgroup 2 analysis
- `Demo0713 sensitivity diabetes diag all.ipynb`: Diabetes diagnosis (any) subgroup analysis
- `Demo0713 sensitivity diabetes diag1.ipynb`: Diabetes diagnosis (primary) subgroup analysis

Each sensitivity analysis notebook includes:
- Subgroup data filtering
- Complete pipeline workflow (feature selection, model training, evaluation)
- Subgroup-specific result analysis and visualization

### 4. Run Individual Modules

You can also run individual modules for testing:

```python
# Data loading
python data_loader.py

# Data preprocessing
python data_preprocessor.py

# Feature selection
python feature_selector.py

# Model training
python model_trainer.py
```

## Pipeline Workflow (Based on Demo0720 main.ipynb)

### Step 1: Data Loading and Preview (`data_loader.py`)
- Load diabetes dataset
- Load ID mapping data
- Merge all data tables
- Generate data summary report

### Step 2: Data Filtering
- **Keep First Admission Records**: Sort by `encounter_id`, keep only the first admission record for each patient
- **Remove Patients Who Cannot Be Readmitted**: Remove records with `discharge_disposition_id` indicating death or hospice care

### Step 3: Data Preprocessing (`data_preprocessor.py`)
- **Feature Engineering**:
  - Create age-related features (age midpoint, age group)
  - Create diagnosis classification features (ICD-9 code classification)
  - Create comorbidity features
  - Create encounter-related features (encounter index, rolling average)
- **Data Cleaning**:
  - Handle missing values (delete columns with >50% missing, fill specified columns with 'Unknown')
  - Handle special characters '?'
- **Data Transformation**:
  - Categorical feature encoding
  - Numerical feature standardization
  - Target variable preparation
- **Data Balancing**:
  - Use SMOTE to handle class imbalance

### Step 4: Exploratory Data Analysis (`eda_analyzer.py`)
- **Basic Statistical Analysis**:
  - Readmission distribution
  - Missing value distribution
  - Average age analysis
- **Medical Interpretation Analysis**:
  - Readmission rates by age group and gender
  - Relationship between comorbidities and readmission rates
  - Relationship between medication dose changes and readmission rates
- **Visualization**:
  - Correlation heatmap
  - Top 10 diagnosis distribution
  - Length of stay analysis

### Step 5: Feature Selection (`feature_selector.py`)
- **Multiple Top N Value Feature Selection**: Support different feature counts like 5, 10, 15
- **Feature Selection Methods**:
  - L1 regularization feature selection
  - Mutual information feature selection
  - Tree model feature importance
- **Result Display**:
  - Feature selection matrix visualization
  - Detailed feature selection tables
  - Summary statistics

### Step 6: Model Training (`model_trainer.py`)
- **Model Types**:
  - Logistic Regression
  - Random Forest
  - XGBoost
- **Hyperparameter Tuning**:
  - GridSearchCV grid search
  - 5-fold cross-validation
  - AUC score-based optimization
- **Evaluation Methods**:
  - Validation set evaluation
  - Test set evaluation
- **Performance Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - AUC-ROC

## Output Files

After running the pipeline, the following files will be generated in the `outputs/` directory:

### Data Files
- `merged_data.csv`: Merged raw data
- `X_train.csv`, `X_val.csv`, `X_test.csv`: Preprocessed feature data
- `y_train.csv`, `y_val.csv`, `y_test.csv`: Target variable data

### Model Files
- `models/LogisticRegression.joblib`: Logistic Regression model
- `models/RandomForest.joblib`: Random Forest model
- `models/XGBoost.joblib`: XGBoost model

### Feature Selection Results
- `selected_features_top15.json`: Selected feature list
- `feature_importance.png`: Feature importance visualization

### Reports and Visualizations
- `model_report.txt`: Model training report
- `model_comparison.png`: Model performance comparison plot
- `final_pipeline_report.txt`: Complete pipeline report
- `pipeline.log`: Detailed execution log

### Excel Result Files
- `0720_15_all_param_search_results_gridcv final_cv5.xlsx`: GridSearchCV detailed results
  - Contains validation set, test set, and cross-validation results for each feature selection method
  - Performance comparison of all hyperparameter combinations

## Configuration Options

The following configurations can be modified in `pipeline_config.py`:

```python
MODEL_CONFIG = {
    'test_size': 0.2,           # Test set proportion
    'val_size': 0.2,            # Validation set proportion
    'random_state': 42,         # Random seed
    'cv_folds': 5,              # Cross-validation folds
    'feature_selection_top_n': 15  # Feature selection count
}
```

## Feature Categories

Features in the project are organized into the following categories:

- **Demographic**: Demographic features (age, gender, race, etc.)
- **Administrative**: Administrative features (admission type, discharge disposition, etc.)
- **Clinical**: Clinical features (diagnosis, comorbidities, etc.)
- **Utilization**: Utilization features (length of stay, number of tests, etc.)
- **Medication**: Medication features (various diabetes medications)

## Model Performance

Based on test set evaluation, performance of different feature selection methods and model combinations using Top-15 features:

### AUC Performance Comparison

| Feature Selection Method | Logistic Regression | Random Forest | XGBoost |
|-------------------------|-------------------|---------------|---------|
| L1 | 0.605 | 0.601 | **0.639** |
| MutualInfo | 0.574 | 0.560 | 0.576 |
| TreeImportance | 0.577 | 0.591 | 0.606 |

### F1-Score Performance Comparison

| Feature Selection Method | Logistic Regression | Random Forest | XGBoost |
|-------------------------|-------------------|---------------|---------|
| L1 | **0.195** | 0.040 | 0.014 |
| MutualInfo | 0.182 | 0.003 | 0.005 |
| TreeImportance | 0.182 | 0.012 | 0.006 |

### Performance Summary

- **Best AUC Performance**: L1 feature selection + XGBoost model (0.639)
- **Best F1-Score Performance**: L1 feature selection + Logistic Regression model (0.195)
- **Overall Performance**: Logistic Regression performs better in F1-Score, while XGBoost performs best in AUC
- **Feature Selection Impact**: L1 regularization feature selection method performs best in most cases

## Sensitivity Analysis

Sensitivity analysis is performed through separate notebooks, with each subgroup analysis including:

### Subgroup Definitions
- **Comorbidity Subgroup 1**: Patients with comorbidity count = 1
- **Comorbidity Subgroup 2**: Patients with comorbidity count ≥ 2
- **Diabetes Diagnosis (Any)**: Patients with any of the three diagnoses being diabetes
- **Diabetes Diagnosis (Primary)**: Patients with primary diagnosis being diabetes

### Analysis Workflow
Each sensitivity analysis notebook executes the complete pipeline:
1. Subgroup data filtering
2. Feature engineering and preprocessing
3. Feature selection
4. Model training and evaluation
5. Result analysis and visualization

### Result Comparison
- Model performance comparison across subgroups
- Feature selection difference analysis
- Medical interpretation and clinical significance

## Extension and Customization

### Adding New Feature Selection Methods

Add new methods in `feature_selector.py`:

```python
def select_features_by_new_method(self, X, y, top_n=15):
    # Implement new feature selection logic
    pass
```

### Adding New Models

Add in the `get_models()` method of `model_trainer.py`:

```python
def get_models(self):
    return {
        # Existing models...
        'NewModel': NewModelClass(random_state=self.random_state)
    }
```

### Modifying Feature Engineering

Add new feature engineering methods in `data_preprocessor.py`:

```python
def create_new_feature(self, df):
    # Implement new feature creation logic
    return df
```

### Adding New Sensitivity Analysis Subgroups

Create new notebook files, such as `Demo0713 sensitivity new_subgroup.ipynb`, including:
- Subgroup definition logic
- Complete pipeline workflow
- Subgroup-specific result analysis

## Troubleshooting

### Common Issues

1. **Insufficient Memory**: For large datasets, reduce the number of feature selections or use data sampling
2. **Dependency Package Version Conflicts**: Use virtual environment and install strictly according to `requirements.txt`
3. **Missing Data Files**: Ensure all required data files are in the correct location

### Log Files

Check the `pipeline.log` file for detailed execution information and error messages.

## Contributing

Welcome to submit issue reports and feature requests. If you want to contribute code:

1. Fork the project
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License.

## Contact

For questions or suggestions, please contact through:
- Submit GitHub Issue
- Send email to project maintainer

---

**Note**: This is a project for educational and research purposes. In actual medical applications, please ensure compliance with relevant medical data privacy regulations and ethical guidelines.
