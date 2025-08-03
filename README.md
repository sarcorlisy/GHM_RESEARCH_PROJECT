# Hospital Readmission Prediction Pipeline

A comprehensive data science pipeline for predicting hospital readmission risk within 30 days of discharge.

## Project Summary

This repository contains a complete hospital readmission prediction system with two distinct branches demonstrating different aspects of data science and data engineering:

### Main Branch - Machine Learning Pipeline
**Focus**: Complete data science workflow from raw data to trained models
- **Data Science**: Feature engineering, model training, and evaluation
- **Machine Learning**: Multiple algorithms (Logistic Regression, Random Forest, XGBoost)
- **Research**: Exploratory data analysis, feature selection, and model comparison
- **Output**: Trained models, performance metrics, and visualization reports

### SQL-Azure-Pipeline Branch - Data Engineering Pipeline
**Focus**: Enterprise-grade data processing and ETL workflows
- **Data Engineering**: ETL pipeline with Azure Data Lake integration
- **Database Management**: MySQL-based data processing and storage
- **Cloud Integration**: Azure Blob Storage for data backup and sharing
- **Production Ready**: Modular architecture with comprehensive logging and error handling

### Key Differences

| Aspect | Main Branch | SQL-Azure-Pipeline Branch |
|--------|-------------|---------------------------|
| **Primary Focus** | Machine Learning & Model Training | Data Engineering & ETL |
| **Data Processing** | Pandas-based in-memory processing | SQL-based database processing |
| **Storage** | Local CSV files | Azure Data Lake + MySQL |
| **Architecture** | Notebook-centric approach | Enterprise ETL pipeline |
| **Use Case** | Research & Prototyping | Production & Enterprise |
| **Skills Demonstrated** | Data Science, ML Modeling | Data Engineering, ETL, Cloud |

## Project Overview

This project demonstrates comprehensive capabilities in healthcare data infrastructure development and research translation, directly aligning with healthcare sector requirements:

### Healthcare Data Infrastructure & Research Translation
- **Independent Research Data Infrastructure**: Complete ETL pipeline with Azure Data Lake integration for scalable healthcare data processing
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
- **Advanced SQL Programming**: Complex T-SQL queries for data transformation, analysis, and optimization
- **Azure Data Lake Integration**: Direct experience with Azure Blob Storage and cloud-based data processing
- **Large-Scale Dataset Handling**: Processing of 71,518+ patient records with complex healthcare data structures
- **Healthcare Domain Expertise**: Deep understanding of medical data structures, ICD-9 codes, and patient records

### Documentation & Transparency
- **Comprehensive Documentation**: Detailed documentation of all data analysis processes, methodologies, and results
- **Reproducible Workflows**: Version-controlled code and configuration files ensuring research replicability
- **Transparent Reporting**: Automated report generation with clear insights and recommendations for operational teams

This project transforms original Jupyter notebooks into a modular, reusable data science pipeline with the following key capabilities:

- **Data Loading & Merging**: Automated loading and merging of multiple data sources
- **Data Preprocessing**: Feature engineering, data cleaning, encoding, and standardization
- **Feature Selection**: Multiple feature selection methods (L1 regularization, mutual information, tree model importance)
- **Model Training**: Multiple machine learning models (Logistic Regression, Random Forest, XGBoost)
- **Model Evaluation**: Cross-validation, test set evaluation, performance comparison
- **Result Visualization**: Feature importance plots, model comparison charts
- **Report Generation**: Automated generation of detailed training and evaluation reports

## Project Structure

```
rp0609/
├── src/                        # Source code directory
│   ├── etl/                    # ETL pipeline modules
│   │   ├── etl_pipeline_new.py    # Main ETL orchestrator
│   │   ├── dynamic_column_cleaner.py  # Dynamic column cleaning
│   │   ├── data_pre_cleaner.py    # Data pre-cleaning module
│   │   └── sql_processing/     # SQL processing scripts
│   ├── api_integration/        # API integration modules
│   ├── utils/                  # Utility modules
│   ├── data_ingestion/         # Data ingestion modules
│   └── ml_pipeline/            # Machine learning pipeline modules
├── config/                     # Configuration files
│   ├── database_config.yaml       # Database configuration
│   ├── azure_config.yaml          # Azure storage configuration
│   └── api_config.yaml            # API configuration
├── docs/                       # Documentation
├── logs/                       # Log files
├── database/                   # Database related files
├── tests/                      # Test files
├── pipeline_config.py             # Pipeline configuration
├── data_loader.py                 # Data loading module
├── data_preprocessor.py           # Data preprocessing module
├── feature_selector.py            # Feature selection module
├── main_pipeline.py               # Main pipeline orchestrator
├── run_complete_etl_pipeline.py   # One-click ETL pipeline
├── run_api_integration.py         # API integration runner
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

## Quick Start

### 1. Clone the Project
```bash
git clone <repository-url>
cd rp0609
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
Create a `.env` file in the project root:
```env
DB_HOST=localhost
DB_PORT=3306
DB_NAME=hospital_readmission
DB_USER=root
DB_PASSWORD=your_password
AZURE_CONNECTION_STRING=your_azure_connection_string
```

### 4. Prepare Data Files
Ensure the following data files are available:
- `diabetic_data.csv` - Main hospital dataset
- `IDS_mapping.csv` - ID mapping data
- `ccs_icd9_mapping.csv` - ICD-9 mapping data

## Usage Guide

### Option 1: Complete ETL Pipeline (Recommended)
```bash
python run_complete_etl_pipeline.py
```
**Features**: Download from Azure → Preprocess → Import to SQL → Mapping → Cleaning → Business cleaning → Upload back to Azure

### Option 2: Individual Modules
```bash
# Data preprocessing
python data_preprocessor.py

# Feature selection
python feature_selector.py

# Complete ML pipeline
python main_pipeline.py

# API integration
python run_api_integration.py
```

### Option 3: ETL Pipeline Steps
```bash
# Run specific ETL steps
python src/etl/etl_pipeline_new.py
```

## Pipeline Workflow

### Phase 1: Data Import & Preprocessing
- **Download from Azure**: Raw data retrieval
- **8-Step Data Cleaning**:
  1. Handle missing values
  2. Handle special characters ('?' → 'Unknown')
  3. Standardize patient IDs
  4. Process age fields (extract midpoints)
  5. Process numeric fields
  6. Standardize medication fields
  7. Process diagnosis fields
  8. Add timestamps
- **Import to MySQL**: Store in `patients` table

### Phase 2: Data Mapping
- **Download mapping data** from Azure
- **Create mapping tables**:
  - `admission_type_mapping`
  - `discharge_disposition_mapping`
  - `admission_source_mapping`
- **Create enriched data** in `patients_mapped` table
- **Upload to Azure** for backup

### Phase 3: Dynamic Column Cleaning
- **Analyze invalid values** in each column
- **Remove high invalid rate columns** (>50% invalid values)
- **Create cleaned table** `patients_cleaned`
- **Upload to Azure**

### Phase 4: Business Rule Cleaning
- **Apply business rules** to filter records
- **Create business cleaned table** `patients_business_cleaned`
- **Upload to Azure**

### Phase 5: Feature Engineering
- **Create ML features** for model training
- **Create features table** `patients_features`
- **Upload to Azure**

## Data Flow

```
Azure (raw-data) 
    ↓
patients (MySQL) 
    ↓
patients_mapped (MySQL + Azure)
    ↓
patients_cleaned (MySQL + Azure) - Dynamic column cleaning
    ↓
patients_business_cleaned (MySQL + Azure) - Business rules
    ↓
patients_features (MySQL + Azure) - Feature engineering
    ↓
ML Pipeline (Python) - Model training & evaluation
```

## Feature Categories

The project organizes features into the following categories:

- **Demographic**: Age, gender, race, etc.
- **Administrative**: Admission type, discharge disposition, etc.
- **Clinical**: Diagnoses, comorbidities, etc.
- **Utilization**: Length of stay, number of procedures, etc.
- **Medication**: Diabetes medications, medication changes, etc.

## Model Performance

Typical performance metrics on test set:

| Model | AUC | F1-Score | Accuracy | Precision | Recall |
|-------|-----|----------|----------|-----------|--------|
| Logistic Regression | 0.670 | 0.580 | 0.65 | 0.62 | 0.55 |
| Random Forest | 0.965 | 0.933 | 0.94 | 0.94 | 0.93 |
| XGBoost | 0.958 | 0.931 | 0.93 | 0.93 | 0.93 |

## Output Files

After running the pipeline, the following files are generated:

### Data Files
- `merged_data.csv`: Merged raw data
- `X_train.csv`, `X_val.csv`, `X_test.csv`: Preprocessed feature data
- `y_train.csv`, `y_val.csv`, `y_test.csv`: Target variable data

### Model Files
- `models/LogisticRegression.joblib`: Logistic regression model
- `models/RandomForest.joblib`: Random forest model
- `models/XGBoost.joblib`: XGBoost model

### Reports & Visualizations
- `model_report.txt`: Model training report
- `model_comparison.png`: Model performance comparison
- `feature_importance.png`: Feature importance visualization
- `pipeline.log`: Detailed execution log

## Configuration

Modify settings in `pipeline_config.py`:

```python
MODEL_CONFIG = {
    'test_size': 0.2,           # Test set ratio
    'val_size': 0.2,            # Validation set ratio
    'random_state': 42,         # Random seed
    'cv_folds': 5,              # Cross-validation folds
    'feature_selection_top_n': 15  # Number of selected features
}
```

## Customization

### Adding New Feature Selection Methods
```python
def select_features_by_new_method(self, X, y, top_n=15):
    # Implement new feature selection logic
    pass
```

### Adding New Models
```python
def get_models(self):
    return {
        # Existing models...
        'NewModel': NewModelClass(random_state=self.random_state)
    }
```

### Adding New Features
```python
def create_new_feature(self, df):
    # Implement new feature creation logic
    return df
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: For large datasets, reduce feature selection count or use data sampling
2. **Package Conflicts**: Use virtual environment and follow `requirements.txt` strictly
3. **Missing Data Files**: Ensure all required data files are in correct locations
4. **Database Connection**: Check database credentials and connection settings

### Logs
Check `pipeline.log` for detailed execution information and error messages.

## Documentation

- **USAGE_GUIDE.md**: Detailed usage instructions
- **CORRECTED_DATA_FLOW.md**: Data flow documentation
- **pipeline_structure.md**: Architecture overview
- **docs/**: Additional documentation

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

`#data-science` `#machine-learning` `#healthcare` `#etl-pipeline` `#python` `#sql` `#azure`
