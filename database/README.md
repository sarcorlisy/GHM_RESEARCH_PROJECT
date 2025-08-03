# Database Extension Module

This module extends the hospital readmission prediction pipeline with comprehensive database capabilities, demonstrating SQL skills, data migration, and analytics operations.

## Overview

The database extension provides:
- **MySQL Database Connectivity**: Full database operations with mysql-connector-python and SQLAlchemy
- **Advanced SQL Queries**: Predefined analytics queries for healthcare data
- **Data Migration Tools**: CSV to database migration with validation
- **Model Result Storage**: Persistent storage of ML model results
- **Analytics Dashboard**: Database-driven analytics and reporting

## Features

### 🗄️ Database Management
- **MySQLConnector**: Core MySQL database connection and operations (in mysql_connector.py)
- **MySQLManager**: High-level MySQL database management interface (in mysql_connector.py)
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

### 1. Setup MySQL Database
```python
# Import MySQL classes directly from the module
from database.mysql_connector import MySQLManager

# Initialize MySQL database
mysql_manager = MySQLManager()
mysql_manager.initialize_database()
```

### 2. Migrate Data
```python
from database import DataMigrator

# Migrate CSV to MySQL database
migrator = DataMigrator(mysql_manager)
data_files = {'patients': 'diabetic_data.csv'}
results = migrator.migrate_all_data(data_files)
```

### 3. Execute Analytics
```python
from database import QueryExecutor

# Run analytics queries
query_executor = QueryExecutor(mysql_manager.connector)
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
mysql_manager.connector.save_model_result(model_result)
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
WHERE created_at >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
GROUP BY DATE(created_at), model_name, feature_selection_method
ORDER BY run_date DESC, avg_auc_score DESC
```

## Database Schema

### Patients Table
```sql
CREATE TABLE patients (
    id INT AUTO_INCREMENT PRIMARY KEY,
    encounter_id VARCHAR(50),
    patient_id VARCHAR(50) UNIQUE NOT NULL,
    race VARCHAR(50),
    gender VARCHAR(20),
    age VARCHAR(10),
    weight VARCHAR(10),
    admission_type_id INT,
    discharge_disposition_id INT,
    admission_source_id INT,
    time_in_hospital INT,
    payer_code VARCHAR(10),
    medical_specialty VARCHAR(50),
    num_lab_procedures INT,
    num_procedures INT,
    num_medications INT,
    number_outpatient INT,
    number_emergency INT,
    number_inpatient INT,
    diag_1 VARCHAR(10),
    diag_2 VARCHAR(10),
    diag_3 VARCHAR(10),
    number_diagnoses INT,
    max_glu_serum VARCHAR(10),
    A1Cresult VARCHAR(10),
    metformin VARCHAR(10),
    repaglinide VARCHAR(10),
    nateglinide VARCHAR(10),
    chlorpropamide VARCHAR(10),
    glimepiride VARCHAR(10),
    acetohexamide VARCHAR(10),
    glipizide VARCHAR(10),
    glyburide VARCHAR(10),
    tolbutamide VARCHAR(10),
    pioglitazone VARCHAR(10),
    rosiglitazone VARCHAR(10),
    acarbose VARCHAR(10),
    miglitol VARCHAR(10),
    troglitazone VARCHAR(10),
    tolazamide VARCHAR(10),
    examide VARCHAR(10),
    citoglipton VARCHAR(10),
    insulin VARCHAR(10),
    `glyburide-metformin` VARCHAR(10),
    `glipizide-metformin` VARCHAR(10),
    `glimepiride-pioglitazone` VARCHAR(10),
    `metformin-rosiglitazone` VARCHAR(10),
    `metformin-pioglitazone` VARCHAR(10),
    change VARCHAR(10),
    diabetesMed VARCHAR(10),
    readmitted VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

### Model Results Table
```sql
CREATE TABLE model_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    model_name VARCHAR(50),
    feature_selection_method VARCHAR(50),
    top_n_features INT,
    accuracy DECIMAL(5,4),
    precision_score DECIMAL(5,4),
    recall DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    auc_score DECIMAL(5,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

## Configuration

### Environment Variables
```bash
DB_HOST=localhost
DB_PORT=3306
DB_NAME=hospital_readmission
DB_USER=root
DB_PASSWORD=hospital123
```

### MySQL Configuration
```python
mysql_config = {
    'host': 'localhost',
    'port': 3306,
    'database': 'hospital_readmission',
    'user': 'root',
    'password': 'hospital123',
    'charset': 'utf8mb4',
    'autocommit': True
}
```

## Usage Examples

### Complete Analytics Workflow
```python
from database.mysql_connector import MySQLManager
from database import QueryExecutor, DataMigrator

# 1. Setup MySQL database
mysql_manager = MySQLManager()
mysql_manager.initialize_database()

# 2. Migrate data
migrator = DataMigrator(mysql_manager)
migrator.migrate_csv_to_database('diabetic_data.csv', 'patients')

# 3. Run analytics
query_executor = QueryExecutor(mysql_manager.connector)
all_analytics = query_executor.execute_all_analytics()

# 4. Generate reports
for name, df in all_analytics.items():
    df.to_excel(f'outputs/{name}_analysis.xlsx', index=False)

mysql_manager.close()
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

### MySQL Connection Example
```python
from database.mysql_connector import MySQLConnector

# Create MySQL connector
mysql_connector = MySQLConnector({
    'host': 'localhost',
    'port': 3306,
    'database': 'hospital_readmission',
    'user': 'root',
    'password': 'hospital123'
})

# Connect to database
if mysql_connector.connect():
    # Execute query
    result = mysql_connector.execute_query("SELECT COUNT(*) FROM patients")
    print(f"Total patients: {result.iloc[0, 0]}")
    
    # Disconnect
    mysql_connector.disconnect()
```

## Available Classes

### From Main Module (`from database import ...`)
- `HospitalReadmissionQueries` - SQL query definitions
- `QueryExecutor` - Query execution interface
- `DataValidator` - Data validation utilities
- `DataTransformer` - Data transformation utilities
- `DataMigrator` - Data migration workflow

### From MySQL Module (`from database.mysql_connector import ...`)
- `MySQLConnector` - MySQL database connector
- `MySQLManager` - MySQL database manager

## Output Files

The database extension generates:
- `database_migration_report.txt`: Detailed migration logs
- `database_analytics_results.xlsx`: Excel file with all analytics results
- MySQL database tables with persistent data storage

## Skills Demonstrated

This database extension showcases:
- **SQL Proficiency**: Complex queries, joins, aggregations, window functions
- **Database Design**: Schema design, normalization, indexing considerations
- **ETL Processes**: Data validation, transformation, loading
- **Data Analytics**: Healthcare-specific analytics and reporting
- **Python Database Integration**: mysql-connector-python, SQLAlchemy, pandas integration
- **Error Handling**: Comprehensive error handling and logging
- **Documentation**: Clear code documentation and usage examples

## Integration with Main Pipeline

The database extension integrates seamlessly with the existing pipeline:
- Extends existing data loading capabilities
- Provides persistent storage for model results
- Enables advanced analytics beyond the original scope
- Maintains compatibility with existing modules

This demonstrates the ability to enhance existing systems with database capabilities while maintaining code quality and documentation standards. 