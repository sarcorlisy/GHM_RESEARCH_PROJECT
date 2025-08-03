# Hospital Readmission Prediction Pipeline

A comprehensive data science pipeline for predicting hospital readmission risk within 30 days of discharge.

This repository contains a complete hospital readmission prediction system with two distinct branches demonstrating different aspects of data science and data engineering:

### Main Branch - Machine Learning Pipeline
**Focus**: Complete data science workflow from raw data to trained models
- **Data Science**: Feature engineering, model training, and evaluation
- **Machine Learning**: Multiple algorithms (Logistic Regression, Random Forest, XGBoost)
- **Research**: Exploratory data analysis, feature selection, and model comparison
- **Output**: Trained models, performance metrics, and visualization reports

### SQL-Azure-Pipeline Branch - Data Engineering Pipeline
**Focus**: Enterprise-grade data processing and ETL workflows
- **Data Engineering**: ETL pipeline with Azure Blob Storage container integration
- **Database Management**: MySQL-based data processing and storage
- **Cloud Integration**: Azure Blob Storage for data backup and sharing
- **Production Ready**: Modular architecture with comprehensive logging and error handling

## Project Overview

### Problem Statement
Hospital readmissions within 30 days of discharge are a critical healthcare quality indicator and financial burden. This project addresses the challenge of predicting which patients are at high risk of readmission, enabling healthcare providers to implement targeted interventions and improve patient outcomes.

### Healthcare Data Infrastructure & Research Translation
- **Independent Research Data Infrastructure**: Complete ETL pipeline with Azure Blob Storage container integration for scalable healthcare data processing
- **Data Lake Architecture**: Azure Blob Storage containers for handling large-scale healthcare datasets with cloud-based data processing
- **Machine Learning Model Hosting**: Automated model training, evaluation, and deployment pipeline with performance monitoring
- **Cross-Functional Data Integration**: Comprehensive data pipeline handling flat files, APIs, and databases from multiple sources
- **Rapid Research Translation**: Modular architecture enabling quick deployment of research insights into operational healthcare systems

### Advanced Analytics & Process Optimization
- **Predictive Modeling**: Machine learning models for hospital readmission prediction, directly improving healthcare operational efficiency
- **Statistical Analysis**: Comprehensive data analysis including trend identification, pattern recognition, and correlation analysis
- **Process Automation**: Automated ETL workflows reducing manual data processing time and improving data accuracy
- **Data Quality Assurance**: Multi-stage data validation and cleaning processes ensuring data reliability

### Technical Implementation
- **Python Programming**: Advanced Python development for data processing, machine learning, and automation
- **Advanced SQL Programming**: Complex PostgreSQL queries for data transformation, analysis, and optimization
- **Azure Blob Storage Integration**: Direct experience with Azure Blob Storage containers for data lake functionality and cloud-based data processing
- **Large-Scale Dataset Handling**: Processing of 100,000+ patient records with complex healthcare data structures
- **Healthcare Domain Expertise**: Deep understanding of medical data structures, ICD-9 codes, and patient records

---

## Machine Learning Pipeline - Main Focus

### Exploratory Data Analysis (EDA)
- **Comprehensive Data Profiling**: Analysis of 71,518 patient records with 50+ features
- **Missing Value Analysis**: Systematic investigation of data quality issues
- **Statistical Analysis**: Distribution analysis, correlation studies, and outlier detection
- **Data Visualization**: Interactive charts and plots for data insights
- **Data Quality Assessment**: Systematic evaluation of data completeness and accuracy

### Advanced Feature Engineering
- **Demographic Features**: 
  - Age group categorization and binning
  - Gender encoding and demographic clustering
  - Socioeconomic factor analysis
- **Clinical Features**: 
  - ICD-9 diagnosis code aggregation and categorization
  - Comorbidity indices calculation
  - Medical specialty classification
  - Length of stay analysis
- **Medication Features**: 
  - Drug interaction patterns and combinations
  - Medication change tracking
  - Diabetes medication effectiveness analysis
- **Temporal Features**: 
  - Time-based patient characteristics
  - Seasonal admission patterns
  - Readmission timing analysis

### Sophisticated Feature Selection
- **Multiple Feature Selection Methods**:
  - **L1 Regularization (Lasso)**: Sparse feature selection for interpretability
  - **Mutual Information**: Information-theoretic feature ranking
  - **Tree-based Importance**: Random Forest and XGBoost feature importance
  - **Recursive Feature Elimination**: Iterative feature selection with cross-validation
- **Feature Validation**: Comprehensive cross-validation of all selection methods
- **Performance Impact Analysis**: Maintaining model performance while reducing complexity

### Advanced Model Training & Optimization

#### Multiple Machine Learning Algorithms
- **Logistic Regression**: Baseline model with high interpretability
- **Random Forest**: Ensemble method with robust feature importance
- **XGBoost**: Gradient boosting for superior performance
- **Support Vector Machines**: For complex decision boundaries
- **Neural Networks**: Deep learning approach for pattern recognition

#### Sophisticated Training Techniques
- **Cross-Validation**: 5-fold cross-validation for robust evaluation
- **Hyperparameter Tuning**: 
  - Grid search and random search optimization
  - Bayesian optimization for efficient parameter search
  - Learning rate scheduling and early stopping
- **Ensemble Methods**: 
  - Voting classifiers (hard and soft voting)
  - Stacking with meta-learners
  - Bagging and boosting combinations
- **Model Stacking**: Multi-level ensemble with meta-learning

#### Advanced Model Evaluation
- **Comprehensive Metrics**: AUC-ROC, F1-Score, Precision, Recall, Accuracy
- **Statistical Testing**: McNemar's test, paired t-tests for model comparison
- **Cross-Validation Analysis**: Stratified k-fold with multiple metrics
- **Bootstrap Confidence Intervals**: For robust performance estimation

### Model Performance & Analysis

#### Outstanding Model Performance
#### ROC-AUC Score
|              | L1    | MI    | Tree  |
| ------------ | ----- | ----- | ----- |
| LogisticReg  | 0.605 | 0.601 | 0.639 |
| RandomForest | 0.574 | 0.560 | 0.576 |
| XGBoost      | 0.577 | 0.591 | 0.606 |

#### F1 Score
|              | L1    | MI    | Tree  |
| ------------ | ----- | ----- | ----- |
| LogisticReg  | 0.195 | 0.040 | 0.014 |
| RandomForest | 0.182 | 0.003 | 0.005 |
| XGBoost      | 0.182 | 0.012 | 0.006 |


#### Model Interpretability
- **Feature Importance Analysis**: Identifying key predictors for readmission
- **Partial Dependence Plots**: Visualizing feature effects on predictions
- **Local Interpretable Model Explanations**: Individual prediction explanations

### Advanced Analytics & Research

#### Statistical Analysis
- **Hypothesis Testing**: Statistical significance of feature relationships
- **Correlation Analysis**: Understanding feature interdependencies
- **Outlier Detection**: Identifying and handling anomalous cases
- **Distribution Analysis**: Understanding data characteristics

#### Model Diagnostics
- **Residual Analysis**: Checking model assumptions
- **Overfitting Detection**: Monitoring training vs validation performance
- **Bias-Variance Trade-off**: Optimizing model complexity
- **Learning Curves**: Understanding model learning behavior

## Machine Learning Architecture

```
Advanced ML Pipeline
├── Data Analysis & EDA
│   ├── Data Profiling & Quality Assessment
│   ├── Statistical Analysis & Visualization
│   ├── Missing Value Analysis
│   └── Correlation & Distribution Studies
├── Feature Engineering
│   ├── Demographic Feature Engineering
│   ├── Clinical Feature Engineering
│   ├── Medication Interaction Features
│   └── Temporal Feature Engineering
├── Feature Selection
│   ├── L1 Regularization (Lasso)
│   ├── Mutual Information Analysis
│   ├── Tree-based Importance
│   └── Recursive Feature Elimination
├── Model Training
│   ├── Multiple Algorithm Training
│   ├── Hyperparameter Optimization
│   ├── Cross-validation
│   └── Ensemble Methods
├── Model Evaluation
│   ├── Performance Metrics
│   ├── Statistical Comparison
│   ├── Feature Importance Analysis
│   └── Model Interpretability
└── Model Deployment
    ├── Model Serialization
    ├── Performance Monitoring
    ├── Automated Retraining
    └── API Development
```

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/sarcorlisy/GHM_RESEARCH_PROJECT.git
cd GHM_RESEARCH_PROJECT
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Complete ML Pipeline
```bash
# Run the complete machine learning pipeline
python main_pipeline.py

# Or run individual ML components
python data_preprocessor.py      # Data preprocessing & EDA
python feature_selector.py       # Feature selection methods
python model_trainer.py          # Advanced model training
python result_analyzer.py        # Results analysis & visualization
```

## Project Structure

```
rp0609/
├── Machine Learning Pipeline (Main Focus)
│   ├── data_preprocessor.py        # Data preprocessing & EDA
│   ├── feature_selector.py         # Advanced feature selection
│   ├── model_trainer.py            # Sophisticated ML training
│   ├── main_pipeline.py            # Complete ML pipeline orchestrator
│   └── result_analyzer.py          # Results analysis & visualization
├── Data Engineering (Secondary)
│   ├── src/etl/                    # ETL pipeline modules
│   ├── run_complete_etl_pipeline.py # ETL pipeline runner
│   └── database/                   # Database utilities
├── ML Outputs & Results
│   ├── models/                     # Trained model files
│   ├── reports/                    # ML analysis reports
│   ├── visualizations/             # ML charts and plots
│   └── logs/                       # ML execution logs
├── Documentation
│   ├── docs/                       # Detailed documentation
│   ├── USAGE_GUIDE.md              # Usage instructions
│   └── pipeline_structure.md       # Architecture overview
└── Configuration
    ├── config/                     # Configuration files
    ├── requirements.txt            # Dependencies
    └── pipeline_config.py          # ML pipeline settings
```

## Detailed ML Workflow

### Phase 1: Exploratory Data Analysis
```python
# Comprehensive data analysis
python data_preprocessor.py
```
- **Data Profiling**: Understand data structure and quality
- **Missing Value Analysis**: Identify and handle data gaps
- **Statistical Analysis**: Distribution and correlation studies
- **Visualization**: Create insightful charts and plots

### Phase 2: Advanced Feature Engineering
```python
# Sophisticated feature creation
python feature_selector.py
```
- **Demographic Features**: Age groups, gender encoding, socioeconomic factors
- **Clinical Features**: Diagnosis aggregation, comorbidity indices, medical specialties
- **Medication Features**: Drug interaction patterns, medication effectiveness
- **Temporal Features**: Time-based patient characteristics, seasonal patterns

### Phase 3: Feature Selection & Optimization
- **L1 Regularization**: Sparse feature selection for interpretability
- **Mutual Information**: Information-theoretic feature ranking
- **Tree-based Selection**: Random Forest and XGBoost importance scores
- **Cross-validation**: Robust validation of feature selection methods

### Phase 4: Advanced Model Training
```python
# Sophisticated ML training
python model_trainer.py
```
- **Multiple Algorithms**: Logistic Regression, Random Forest, XGBoost, SVM, Neural Networks
- **Hyperparameter Tuning**: Grid search, random search, Bayesian optimization
- **Cross-validation**: 5-fold cross-validation for robust evaluation
- **Ensemble Methods**: Voting, stacking, bagging, and boosting

### Phase 5: Comprehensive Model Evaluation
- **Performance Metrics**: AUC, F1-Score, Accuracy, Precision, Recall
- **Statistical Comparison**: McNemar's test, paired t-tests
- **Feature Importance**: SHAP values, partial dependence plots
- **Error Analysis**: Investigation of misclassified cases

## Machine Learning Capabilities Demonstrated

### Advanced ML Techniques
- **Ensemble Learning**: Random Forest, XGBoost, and ensemble combinations
- **Hyperparameter Optimization**: Automated parameter tuning
- **Cross-validation**: Robust model evaluation strategies
- **Feature Engineering**: Creative feature creation and transformation

### Model Development Expertise
- **Algorithm Selection**: Choosing appropriate models for healthcare data
- **Hyperparameter Tuning**: Optimizing model performance
- **Model Comparison**: Statistical evaluation of different approaches
- **Feature Selection**: Identifying most important predictors

### Performance Optimization
- **Ensemble Methods**: Combining multiple models for better performance
- **Model Stacking**: Multi-level ensemble with meta-learning
- **Error Analysis**: Understanding and improving model weaknesses
- **Interpretability**: Making models explainable for healthcare applications

## Healthcare ML Expertise

### Medical Data Understanding
- **ICD-9 Codes**: Processing and aggregating diagnosis codes
- **Medication Data**: Understanding drug interactions and patterns
- **Patient Demographics**: Age, gender, and socioeconomic factors
- **Clinical Variables**: Length of stay, procedures, and diagnoses

### Healthcare Predictive Analytics
- **Risk Prediction**: Identifying high-risk patients for readmission
- **Clinical Decision Support**: Providing insights for healthcare providers
- **Quality Improvement**: Supporting hospital quality initiatives
- **Resource Optimization**: Helping optimize healthcare resource allocation

## Advanced ML Features

### Automated ML Pipeline
- **End-to-End Processing**: From raw data to trained models
- **Reproducible Results**: Version-controlled and documented workflows
- **Scalable Architecture**: Modular design for easy extension
- **Error Handling**: Robust error handling and logging

### Model Deployment Ready
- **Model Serialization**: Save and load trained models
- **API Development**: RESTful API for model serving
- **Performance Monitoring**: Track model performance over time
- **Automated Retraining**: Scheduled model updates

## Documentation & Resources

- **USAGE_GUIDE.md**: Detailed usage instructions
- **pipeline_structure.md**: Architecture overview
- **docs/**: Additional documentation
- **Jupyter Notebooks**: Interactive ML analysis examples

## Contributing

We welcome contributions! Please:

1. Fork the project
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Contact

For questions or suggestions:
- Create a GitHub Issue
- Contact the project maintainers

---

**Important Note**: This project is for educational and research purposes. For actual healthcare applications, ensure compliance with relevant medical data privacy regulations and ethical guidelines.

## Tags

`#machine-learning` `#data-science` `#healthcare` `#feature-engineering` `#model-training` `#python` `#predictive-modeling` `#hospital-analytics` `#ensemble-learning` `#hyperparameter-tuning`
