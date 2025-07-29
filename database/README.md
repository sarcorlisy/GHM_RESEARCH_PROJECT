# Database Extension Module

This module extends the hospital readmission prediction pipeline with comprehensive database capabilities, demonstrating SQL skills, data migration, and analytics operations.

## Overview

The database extension provides:
- **PostgreSQL Database Connectivity**: Full database operations with SQLAlchemy
- **Advanced SQL Queries**: Predefined analytics queries for healthcare data
- **Data Migration Tools**: CSV to database migration with validation
- **Model Result Storage**: Persistent storage of ML model results
- **Analytics Dashboard**: Database-driven analytics and reporting

## Features

### 🗄️ Database Management
- **DatabaseConnector**: Core database connection and operations
- **DatabaseManager**: High-level database management interface
- **Table Creation**: Automatic schema creation for healthcare data
- **Connection Pooling**: Efficient database connection management

### 📊 SQL Analytics
- **HospitalReadmissionQueries**: 20+ predefined SQL queries
- **QueryExecutor**: Easy-to-use query execution interface
- **Analytics Categories**:
  - Patient demographics analysis
  - Readmission risk factors
  - Diagnosis-based analysis
  - Medication impact analysis
  - Length of stay analysis
  - Comorbidity analysis
  - Model performance tracking
  - Operational metrics

### 🔄 Data Migration
- **DataValidator**: Comprehensive data validation
- **DataTransformer**: Data cleaning and transformation
- **DataMigrator**: Complete migration workflow
- **Migration Reporting**: Detailed migration logs and reports

## Quick Start

### 1. Setup Database
```python
from database import DatabaseManager

# Initialize database
db_manager = DatabaseManager()
db_manager.initialize_database()
```

### 2. Migrate Data
```python
from database import DataMigrator

# Migrate CSV to database
migrator = DataMigrator(db_manager)
data_files = {'patients': 'diabetic_data.csv'}
results = migrator.migrate_all_data(data_files)
```

### 3. Execute Analytics
```python
from database import QueryExecutor

# Run analytics queries
query_executor = QueryExecutor(db_manager.connector)
demographics = query_executor.execute_demographics_analysis()
risk_factors = query_executor.execute_risk_factors_analysis()
```

### 4. Store Model Results
```python
# Store ML model results
model_result = {
    'model_name': 'LogisticRegression',
    'feature_selection_method': 'L1',
    'top_n_features': 15,
    'accuracy': 0.605,
    'precision': 0.195,
    'recall': 0.182,
    'f1_score': 0.188,
    'auc_score': 0.639
}
db_manager.connector.save_model_result(model_result)
```

## SQL Query Examples

### Patient Demographics Analysis
```sql
SELECT 
    age,
    gender,
    COUNT(*) as patient_count,
    AVG(time_in_hospital) as avg_length_of_stay,
    SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) as early_readmissions,
    ROUND(
        SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    ) as early_readmission_rate
FROM patients 
GROUP BY age, gender 
ORDER BY age, gender
```

### High-Risk Patient Identification
```sql
SELECT 
    patient_id,
    age,
    gender,
    time_in_hospital,
    num_medications,
    CASE 
        WHEN num_medications > 10 THEN 1 ELSE 0
    END + 
    CASE 
        WHEN time_in_hospital > 7 THEN 1 ELSE 0
    END +
    CASE 
        WHEN num_lab_procedures > 20 THEN 1 ELSE 0
    END as risk_score
FROM patients
WHERE (
    CASE WHEN num_medications > 10 THEN 1 ELSE 0 END + 
    CASE WHEN time_in_hospital > 7 THEN 1 ELSE 0 END +
    CASE WHEN num_lab_procedures > 20 THEN 1 ELSE 0 END
) >= 2
ORDER BY risk_score DESC
```

### Model Performance Trends
```sql
SELECT 
    DATE(created_at) as run_date,
    model_name,
    feature_selection_method,
    AVG(auc_score) as avg_auc_score,
    AVG(f1_score) as avg_f1_score,
    COUNT(*) as runs_count
FROM model_results 
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at), model_name, feature_selection_method
ORDER BY run_date DESC, avg_auc_score DESC
```

## Database Schema

### Patients Table
```sql
CREATE TABLE patients (
    patient_id VARCHAR(50) PRIMARY KEY,
    age INTEGER,
    gender VARCHAR(10),
    race VARCHAR(50),
    admission_type_id INTEGER,
    discharge_disposition_id INTEGER,
    admission_source_id INTEGER,
    time_in_hospital INTEGER,
    num_lab_procedures INTEGER,
    num_procedures INTEGER,
    num_medications INTEGER,
    number_outpatient INTEGER,
    number_emergency INTEGER,
    number_inpatient INTEGER,
    diag_1 VARCHAR(10),
    diag_2 VARCHAR(10),
    diag_3 VARCHAR(10),
    readmitted VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Model Results Table
```sql
CREATE TABLE model_results (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50),
    feature_selection_method VARCHAR(50),
    top_n_features INTEGER,
    accuracy DECIMAL(5,4),
    precision DECIMAL(5,4),
    recall DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    auc_score DECIMAL(5,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Configuration

### Environment Variables
```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=hospital_readmission
DB_USER=postgres
DB_PASSWORD=your_password
```

### Database Configuration
```python
db_config = {
    'host': 'localhost',
    'port': '5432',
    'database': 'hospital_readmission',
    'user': 'postgres',
    'password': 'password'
}
```

## Usage Examples

### Complete Analytics Workflow
```python
from database import DatabaseManager, QueryExecutor, DataMigrator

# 1. Setup database
db_manager = DatabaseManager()
db_manager.initialize_database()

# 2. Migrate data
migrator = DataMigrator(db_manager)
migrator.migrate_csv_to_database('diabetic_data.csv', 'patients')

# 3. Run analytics
query_executor = QueryExecutor(db_manager.connector)
all_analytics = query_executor.execute_all_analytics()

# 4. Generate reports
for name, df in all_analytics.items():
    df.to_excel(f'outputs/{name}_analysis.xlsx', index=False)

db_manager.close()
```

### Data Validation
```python
from database import DataValidator

validator = DataValidator()
validation_results = validator.validate_patient_data(df)

if validation_results['is_valid']:
    print("Data validation passed")
else:
    print("Data validation failed:", validation_results)
```

## Output Files

The database extension generates:
- `database_migration_report.txt`: Detailed migration logs
- `database_analytics_results.xlsx`: Excel file with all analytics results
- Database tables with persistent data storage

## Skills Demonstrated

This database extension showcases:
- **SQL Proficiency**: Complex queries, joins, aggregations, window functions
- **Database Design**: Schema design, normalization, indexing considerations
- **ETL Processes**: Data validation, transformation, loading
- **Data Analytics**: Healthcare-specific analytics and reporting
- **Python Database Integration**: SQLAlchemy, psycopg2, pandas integration
- **Error Handling**: Comprehensive error handling and logging
- **Documentation**: Clear code documentation and usage examples

## Integration with Main Pipeline

The database extension integrates seamlessly with the existing pipeline:
- Extends existing data loading capabilities
- Provides persistent storage for model results
- Enables advanced analytics beyond the original scope
- Maintains compatibility with existing modules

This demonstrates the ability to enhance existing systems with database capabilities while maintaining code quality and documentation standards. 