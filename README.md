# Hospital Readmission Prediction Pipeline

A comprehensive hospital readmission prediction data science pipeline for predicting the risk of patients being readmitted within 30 days after discharge.

## Quick Start

**Not sure which file to run** Check [USAGE_GUIDE.md](USAGE_GUIDE.md) for detailed usage instructions!

### Most Common Commands:

```bash
# 1. Complete ETL Pipeline (Recommended - One-click execution)
python run_complete_etl_pipeline.py

# 2. Detailed ETL Pipeline (Advanced - Step-by-step with detailed logging)
python src/etl/etl_pipeline_new.py

# 3. API Integration Demo
python demo_api_integration.py

# 4. Full API Integration Process
python run_api_integration.py --full

# 5. Machine Learning Model Training
python model_trainer.py
```

## Project Overview

This project implements a comprehensive hospital readmission prediction system with the following main features:

- **ETL Pipeline**: Complete data extraction, transformation, and loading from Azure to MySQL
- **Dynamic Data Cleaning**: Automatic removal of columns with high invalid value rates (>50%)
- **API Integration**: Heterogeneous data source integration for data enrichment
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
├── src/                        # Source code directory
│   ├── etl/                    # ETL pipeline modules
│   │   ├── etl_pipeline_new.py # Detailed ETL pipeline (step-by-step with logging)
│   │   ├── dynamic_column_cleaner.py  # Dynamic column cleaning utility
│   │   ├── data_pre_cleaner.py # Data pre-cleaning module
│   │   └── sql_processing/     # SQL processing scripts
│   │       ├── 01_data_mapping.sql
│   │       ├── 03_business_cleaning.sql
│   │       └── 04_feature_engineering.sql
│   ├── api_integration/        # API integration modules
│   │   ├── api_manager.py      # API management and rate limiting
│   │   ├── api_config.py       # API configuration
│   │   └── data_enricher.py    # Data enrichment utilities
│   ├── utils/                  # Utility modules
│   ├── data_ingestion/         # Data ingestion modules
│   └── ml_pipeline/            # Machine learning pipeline modules
├── config/                     # Configuration files
│   ├── database_config.yaml    # Database configuration
│   ├── azure_config.yaml       # Azure storage configuration
│   └── api_config.yaml         # API configuration
├── run_complete_etl_pipeline.py # One-click ETL pipeline runner
├── run_api_integration.py      # API integration runner
├── demo_api_integration.py     # API integration demo
├── model_trainer.py            # Model training module
├── feature_selector.py         # Feature selection module
├── data_preprocessor.py        # Data preprocessing module
├── eda_analyzer.py             # Exploratory data analysis module
├── sensitivity_analyzer.py     # Sensitivity analysis module
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
├── USAGE_GUIDE.md              # Detailed usage guide
├── CORRECTED_DATA_FLOW.md      # Data flow documentation
├── logs/                       # Log files directory
├── database/                   # Database related files
├── docs/                       # Documentation
└── tests/                      # Test files
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

3. **Configure environment variables**
Create a `.env` file in the project root with your database and Azure credentials:
```env
DB_HOST=localhost
DB_PORT=3306
DB_NAME=hospital_readmission
DB_USER=root
DB_PASSWORD=your_password
AZURE_CONNECTION_STRING=your_azure_connection_string
```

4. **Prepare data files**
Ensure the following data files are in Azure storage:
- `raw-data/diabetic_data.csv` - Main dataset
- `raw-data/IDS_mapping.csv` - ID mapping data

## Usage

### 1. ETL Pipeline

#### Option A: One-Click ETL Pipeline (Recommended for most users)
```bash
python run_complete_etl_pipeline.py
```

**Features:**
- **One-click execution**: Complete ETL process in a single command
- **Automatic timestamping**: Creates timestamped output files
- **Comprehensive logging**: Detailed logs saved to `logs/complete_etl_pipeline.log`
- **Error handling**: Robust error handling and recovery
- **Progress tracking**: Real-time progress updates

#### Option B: Detailed ETL Pipeline (For advanced users and debugging)
```bash
python src/etl/etl_pipeline_new.py
```

**Features:**
- **Step-by-step execution**: Detailed control over each ETL step
- **Advanced logging**: Verbose logging for debugging and monitoring
- **Individual step execution**: Can run specific steps independently
- **Detailed error reporting**: Comprehensive error analysis

**Both pipelines execute:**
- **Data Import**: Download from Azure and import to MySQL with 8-step pre-cleaning
- **Data Mapping**: Create mapping tables and enrich data with descriptions
- **Dynamic Column Cleaning**: Automatically remove columns with >50% invalid values
- **Business Cleaning**: Apply business rules to filter records
- **Feature Engineering**: Create machine learning features
- **Azure Upload**: Upload all processed tables to Azure

### 2. API Integration

#### Quick Demo
```bash
python demo_api_integration.py
```

#### Full API Integration
```bash
# Test API connections
python run_api_integration.py --test

# Run diagnosis enrichment
python run_api_integration.py --diagnosis

# Run medication enrichment
python run_api_integration.py --medication

# Run complete enrichment
python run_api_integration.py --full
```

**Available APIs:**
- **ICD-9/ICD-10 API**: Diagnosis code information
- **OpenFDA API**: Drug information
- **FHIR API**: Patient demographic data
- **Data Quality API**: Data validation
- **Heterogeneous APIs**: Lerner Research Institute, External Vendors, Research Teams

### 3. Machine Learning Pipeline

#### Model Training
```bash
python model_trainer.py
```

This will execute:
- Data loading and preprocessing
- Feature selection using multiple methods
- Model training (Logistic Regression, Random Forest, XGBoost)
- Cross-validation and evaluation
- Performance comparison and visualization

#### Feature Selection
```bash
python feature_selector.py
```

#### Exploratory Data Analysis
```bash
python eda_analyzer.py
```

#### Sensitivity Analysis
```bash
python run_sensitivity_analysis.py
```

### 4. Individual Modules

You can also run individual modules for testing:

```bash
# Data preprocessing
python data_preprocessor.py

# Feature selection
python feature_selector.py

# EDA analysis
python eda_analyzer.py
```

## ETL Pipeline Workflow

### Step 1: Data Import (`import_data_from_azure`)
- Download `diabetic_data.csv` from Azure
- Apply 8-step pre-cleaning:
  1. Handle missing values
  2. Handle special characters ('?' → 'Unknown')
  3. Standardize patient IDs
  4. Process age fields (extract midpoints)
  5. Process numeric fields
  6. Standardize medication fields
  7. Process diagnosis fields
  8. Add timestamps
- Import to MySQL `patients` table

### Step 2: Data Mapping (`run_data_mapping`)
- Download `IDS_mapping.csv` from Azure
- Create mapping tables:
  - `admission_type_mapping`
  - `discharge_disposition_mapping`
  - `admission_source_mapping`
- Create `patients_mapped` table with enriched descriptions
- Upload to Azure

### Step 3: Dynamic Column Cleaning (`run_data_cleaning`)
- Analyze invalid values in each column (NULL, 'Unknown', 'Not Available', etc.)
- Identify columns with invalid value rates > 50%
- Dynamically build SQL to exclude problematic columns
- Create `patients_cleaned` table
- Upload to Azure

### Step 4: Business Cleaning (`run_business_cleaning`)
- Apply business rules to filter records
- Create `patients_business_cleaned` table
- Upload to Azure

### Step 5: Feature Engineering (`run_feature_engineering`)
- Create machine learning features
- Create `patients_features` table
- Upload to Azure

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
patients_business_cleaned (MySQL + Azure)
    ↓
patients_features (MySQL + Azure) - ML ready
```

## Key Features

### Dynamic Column Cleaning
- **Automatic Detection**: Identifies columns with >50% invalid values
- **Smart Removal**: Removes problematic columns while preserving data integrity
- **Data Type Handling**: Properly handles 'Unknown' values in numeric columns
- **Retry Mechanism**: Robust error handling with automatic retries

### API Integration
- **Multiple APIs**: ICD-9, OpenFDA, FHIR, Data Quality, Heterogeneous sources
- **Rate Limiting**: Built-in rate limiting and error handling
- **Data Enrichment**: Enhances data with external information
- **Modular Design**: Easy to add new APIs

### Machine Learning
- **Feature Selection**: Multiple methods (L1, Mutual Information, Tree-based)
- **Model Training**: Logistic Regression, Random Forest, XGBoost
- **Cross-validation**: Robust model evaluation
- **Performance Analysis**: Comprehensive performance metrics

### Cloud Integration
- **Azure Storage**: Seamless cloud storage integration
- **Automatic Upload**: All processed tables uploaded to Azure
- **Version Control**: Timestamped file versions
- **Scalable**: Designed for large datasets

## ETL Pipeline Comparison

| Feature | `run_complete_etl_pipeline.py` | `src/etl/etl_pipeline_new.py` |
|---------|--------------------------------|--------------------------------|
| **Execution Style** | One-click, automated | Step-by-step, detailed |
| **Use Case** | Production, regular runs | Development, debugging |
| **Logging** | Comprehensive file logging | Verbose console + file logging |
| **Error Handling** | Automatic recovery | Detailed error analysis |
| **Output Files** | Timestamped automatically | Manual naming |
| **Progress Tracking** | Real-time updates | Step-by-step confirmation |
| **Configuration** | Built-in defaults | Flexible configuration |

## Troubleshooting

### Common Issues

1. **Database Connection Issues**
   - Check MySQL service is running
   - Verify database credentials in `.env` file
   - Ensure database `hospital_readmission` exists

2. **Azure Connection Issues**
   - Verify Azure connection string in `.env` file
   - Check Azure storage account permissions
   - Ensure required containers exist

3. **Dynamic Column Cleaning Errors**
   - The pipeline includes automatic retry mechanisms
   - Check logs for detailed error information
   - Verify `patients_mapped` table exists before running cleaning

### Logs
- Complete ETL logs: `logs/complete_etl_pipeline.log`
- Detailed ETL logs: `logs/etl_pipeline.log`
- API logs: `logs/api_integration.log`
- General logs: `logs/` directory

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
