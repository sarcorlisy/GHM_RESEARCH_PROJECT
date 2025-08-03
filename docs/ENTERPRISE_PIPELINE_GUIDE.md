# Hospital Readmission Prediction - Enterprise Pipeline Guide

## Overview

This document describes the enterprise-grade data processing pipeline for the Hospital Readmission Prediction project. The pipeline demonstrates real-world data engineering practices including:

- **T-SQL Data Processing**: Advanced SQL queries for data cleaning and feature engineering
- **ETL Pipeline**: Automated data extraction, transformation, and loading
- **Azure Data Lake Integration**: Cloud-based data storage and processing
- **Configuration Management**: Environment-specific configurations
- **Logging and Monitoring**: Structured logging for production environments
- **Modular Architecture**: Separation of concerns and reusable components

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw Data      │    │   SQL Processing│    │   Azure Data    │
│   (CSV)         │───▶│   (MySQL)       │───▶│   Lake          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Python ML     │
                       │   Pipeline      │
                       └─────────────────┘
```

## Project Structure

```
hospital_readmission_project/
├── src/
│   ├── data_ingestion/          # Data upload to Azure
│   │   └── azure_uploader.py
│   ├── etl/                     # ETL pipeline
│   │   ├── sql_processing/      # SQL scripts
│   │   │   ├── 01_data_mapping.sql
│   │   │   ├── 02_data_cleaning.sql
│   │   │   ├── 03_business_cleaning.sql
│   │   │   └── 04_feature_engineering.sql
│   │   └── etl_pipeline.py      # Main ETL orchestrator
│   ├── ml_pipeline/             # Machine learning (future)
│   └── utils/                   # Utilities
│       ├── config.py           # Configuration management
│       └── logging_config.py   # Logging setup
├── config/                      # Configuration files
│   ├── database_config.yaml
│   └── azure_config.yaml
├── tests/                       # Unit tests
├── docs/                        # Documentation
├── logs/                        # Log files
├── run_pipeline.py             # Main runner script
└── requirements.txt            # Dependencies
```

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Create logs directory
mkdir -p logs
```

### 2. Configure Azure (Optional)

Edit `config/azure_config.yaml` with your Azure credentials:

```yaml
azure:
  storage_account:
    name: your-storage-account-name
    connection_string: your-connection-string
```

### 3. Run the Pipeline

```bash
# Run full pipeline (development mode)
python run_pipeline.py --config development

# Run with Azure upload
python run_pipeline.py --config development --upload-azure

# Run specific steps
python run_pipeline.py --step cleaning
python run_pipeline.py --step features

# Run with debug logging
python run_pipeline.py --log-level DEBUG
```

## SQL Processing

### Data Mapping (`01_data_mapping.sql`)
- **Purpose**: Create mapping tables and merge with main data
- **Execution Order**: 1st (after Python import with 8 cleaning rules)
- **Input**: `patients` table (with 8 cleaning rules applied)
- **Output**: `patients_mapped` table
- **Key Features**:
  - Creates admission_type_mapping, discharge_disposition_mapping, admission_source_mapping
  - Merges mapping data with patient records
  - Adds descriptive text for ID fields

### Data Cleaning (`02_data_cleaning.sql`)
- **Purpose**: Basic data cleaning and quality improvements
- **Execution Order**: 2nd (after mapping, before dynamic column cleaning in Python)
- **Input**: `patients_mapped` table
- **Output**: `patients_cleaned` table
- **Key Features**:
  - Additional data quality improvements beyond Python 8 rules
  - Creates cleaned versions of all fields
  - Adds data quality flags
  - Note: Dynamic column cleaning done in Python after this step

### Business Rules Cleaning (`03_business_cleaning.sql`)
- **Purpose**: Apply business rules and final data cleaning
- **Execution Order**: 3rd (after dynamic column cleaning in Python)
- **Input**: `patients_cleaned` table (after dynamic column cleaning)
- **Output**: `patients_business_cleaned` table
- **Key Features**:
  - Removes patients who cannot be readmitted (deceased/hospice)
  - Applies business logic filters
  - Creates final cleaned dataset for analysis

### Feature Engineering (`04_feature_engineering.sql`)
- **Purpose**: Create features for machine learning models
- **Execution Order**: 4th (final step, after all cleaning is complete)
- **Input**: `patients_business_cleaned` table
- **Output**: `patients_features` table
- **Key Features**:
  - Creates age groups, stay duration categories
  - Calculates visit frequency and diagnosis complexity
  - Generates risk scores and categories
  - Prepares target variables for ML models

## ETL Pipeline

### Pipeline Components

1. **Data Extraction**: Read from MySQL database
2. **Data Transformation**: Execute SQL processing scripts
3. **Data Loading**: Upload to Azure Data Lake
4. **Quality Control**: Generate data quality reports
5. **Output Generation**: Create files for ML pipeline

### Pipeline Execution

```python
from src.etl.etl_pipeline import HospitalReadmissionETL

# Initialize pipeline
etl = HospitalReadmissionETL()

# Run full pipeline
success = etl.run_full_pipeline(upload_to_azure=True)

# Run individual steps
etl.connect_to_mysql()
etl.run_data_cleaning()
etl.run_feature_engineering()
processed_data = etl.get_processed_data()
etl.disconnect_from_mysql()
```

## Azure Integration

### Data Lake Structure

```
azure-storage-account/
├── raw-data/                    # Original data
│   └── diabetic_data_raw.csv
├── processed-data/              # Processed data
│   ├── hospital_readmission_processed.csv
│   └── hospital_readmission_processed.parquet
└── ml-results/                  # Model outputs (future)
```

### Upload Data

```python
from src.data_ingestion.azure_uploader import AzureDataUploader

uploader = AzureDataUploader()

# Upload CSV file
success = uploader.upload_csv_to_azure(
    local_file_path="diabetic_data.csv",
    container_name="raw-data"
)

# Upload DataFrame
success = uploader.upload_dataframe_to_azure(
    df=processed_data,
    container_name="processed-data",
    blob_name="processed_data.parquet",
    file_format="parquet"
)
```

## Configuration Management

### Environment Configurations

- **Development**: Uses local MySQL, optional Azure
- **Production**: Uses Azure services, full logging

### Configuration Files

- `database_config.yaml`: Database connection settings
- `azure_config.yaml`: Azure service configurations

## Logging and Monitoring

### Log Structure

```
logs/
├── pipeline_development.log
├── pipeline_production.log
└── etl_report.json
```

### Log Levels

- **DEBUG**: Detailed debugging information
- **INFO**: General information about pipeline progress
- **WARNING**: Warning messages for potential issues
- **ERROR**: Error messages for failed operations

## Data Quality

### Quality Metrics

- **Completeness**: Percentage of non-null values
- **Accuracy**: Data validation checks
- **Consistency**: Cross-field validation
- **Timeliness**: Processing time tracking

### Quality Reports

The pipeline generates comprehensive quality reports:

```json
{
  "data_quality": [
    {"metric": "Total Records", "value": 71518},
    {"metric": "Records with Clean Age", "value": 71234},
    {"metric": "Records with Clean Gender", "value": 71456}
  ],
  "feature_summary": [
    {"feature_group": "Age Statistics", "metric": "Mean Age", "value": 54.23},
    {"feature_group": "Readmission Statistics", "metric": "30-Day Readmission Rate", "value": 11.27}
  ],
  "table_counts": {
    "patients": 71518,
    "patients_cleaned": 71518,
    "patients_features": 71518
  }
}
```

## Best Practices

### Code Organization

1. **Separation of Concerns**: Each module has a single responsibility
2. **Configuration Management**: Environment-specific settings
3. **Error Handling**: Comprehensive exception handling
4. **Logging**: Structured logging for monitoring
5. **Testing**: Unit tests for critical components

### Data Processing

1. **Data Validation**: Validate data at each step
2. **Quality Checks**: Monitor data quality metrics
3. **Audit Trail**: Track all data transformations
4. **Performance**: Optimize SQL queries and data processing
5. **Scalability**: Design for large datasets

### Production Deployment

1. **Environment Isolation**: Separate dev/staging/prod environments
2. **Security**: Secure credential management
3. **Monitoring**: Real-time pipeline monitoring
4. **Backup**: Regular data backups
5. **Documentation**: Comprehensive documentation

## Troubleshooting

### Common Issues

1. **MySQL Connection**: Check database credentials and service status
2. **Azure Upload**: Verify Azure credentials and network connectivity
3. **SQL Errors**: Check SQL syntax and database permissions
4. **Memory Issues**: Monitor memory usage for large datasets

### Debug Mode

```bash
# Run with debug logging
python run_pipeline.py --log-level DEBUG

# Check specific components
python -c "from src.etl.etl_pipeline import HospitalReadmissionETL; etl = HospitalReadmissionETL(); print('ETL initialized successfully')"
```

## Future Enhancements

1. **Machine Learning Integration**: Add ML model training and deployment
2. **Real-time Processing**: Implement streaming data processing
3. **Advanced Monitoring**: Add metrics and alerting
4. **API Integration**: Create REST APIs for data access
5. **Containerization**: Docker support for deployment
6. **CI/CD Pipeline**: Automated testing and deployment

## Skills Demonstrated

This pipeline demonstrates the following skills required for the Analytics Engineer position:

- ✅ **Python Programming**: Advanced Python with proper architecture
- ✅ **T-SQL Programming**: Complex SQL queries and data processing
- ✅ **Large-scale Data**: Processing 71,518+ records efficiently
- ✅ **ETL Development**: Complete ETL pipeline implementation
- ✅ **Azure Integration**: Data Lake and cloud services
- ✅ **Healthcare Data**: Medical data processing and analysis
- ✅ **Configuration Management**: Environment-specific settings
- ✅ **Logging and Monitoring**: Production-grade logging
- ✅ **Modular Design**: Reusable and maintainable code
- ✅ **Documentation**: Comprehensive documentation 