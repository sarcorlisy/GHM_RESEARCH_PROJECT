# Corrected Data Flow Documentation

## Actual Data Flow

Based on the actual ETL pipeline code analysis, your complete data flow is:

```
Raw Data (diabetic_data.csv in Azure)
    ↓
Download from Azure
    ↓
patients (MySQL table - First admission filtering + 8-step pre-cleaning)
    ↓
Mapping Tables Creation (admission_type_mapping, discharge_disposition_mapping, admission_source_mapping)
    ↓
patients_mapped (Mapped data - Added description columns)
    ↓
patients_cleaned (Removed ID columns + Dynamic column cleaning)
    ↓
patients_business_cleaned (Business rule filtering - Removed patients who cannot be readmitted)
    ↓
patients_features (Feature engineering - ML ready data)
    ↓
Upload all tables to Azure
```

## Complete ETL Pipeline Steps

### Step 1: Data Import from Azure
- Download `diabetic_data.csv` from Azure `raw-data` container
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

### Step 2: Mapping Tables Creation
- Download `IDS_mapping.csv` from Azure
- Create mapping tables:
  - `admission_type_mapping`
  - `discharge_disposition_mapping`
  - `admission_source_mapping`
- Insert mapping data

### Step 3: Data Mapping
- Create `patients_mapped` table with enriched descriptions
- Join with mapping tables to add descriptive columns
- Upload to Azure as `mapped_data.csv`

### Step 4: Dynamic Column Cleaning
- Analyze invalid values in each column (NULL, 'Unknown', 'Not Available', etc.)
- Identify columns with invalid value rates > 50%
- Dynamically build SQL to exclude problematic columns
- Create `patients_cleaned` table
- Upload to Azure as `patients_cleaned.csv`

### Step 5: Business Cleaning
- Apply business rules to filter records
- Remove patients who cannot be readmitted (death, hospice, etc.)
- Create `patients_business_cleaned` table
- Upload to Azure as `patients_business_cleaned.csv`

### Step 6: Feature Engineering
- Create machine learning features
- Create `patients_features` table
- Upload to Azure

## File Correspondence

| Stage | Table Name | Azure File | Rows | Columns | Description |
|-------|------------|------------|------|---------|-------------|
| Raw | diabetic_data.csv | raw-data/diabetic_data.csv | 101,766 | 50 | Original diabetes data |
| Import | patients | - | 71,518 | 50 | First admission filtering + pre-cleaning |
| Mapping Tables | admission_type_mapping, discharge_disposition_mapping, admission_source_mapping | - | - | - | ID to description mappings |
| Mapping | patients_mapped | processed-data/mapped_data.csv | 71,518 | 53 | +3 mapping description columns |
| Cleaning 1 | patients_cleaned | processed-data/patients_cleaned.csv | 71,518 | 49 | -3 ID columns + dynamic column cleaning |
| Cleaning 2 | patients_business_cleaned | processed-data/patients_business_cleaned.csv | 69,973 | 49 | -1,545 business filtered records |
| Features | patients_features | processed-data/patients_features.csv | 69,973 | 50+ | ML ready features |

## Key Features

### Dynamic Column Cleaning
- **Automatic Detection**: Identifies columns with >50% invalid values
- **Smart Removal**: Removes problematic columns while preserving data integrity
- **Data Type Handling**: Properly handles 'Unknown' values in numeric columns
- **Retry Mechanism**: Robust error handling with automatic retries

### Azure Integration
- **Automatic Download**: Downloads raw data from Azure
- **Automatic Upload**: All processed tables uploaded to Azure
- **Version Control**: Timestamped file versions
- **Container Organization**: 
  - `raw-data/` - Original data files
  - `processed-data/` - Processed data files

### Error Handling
- **Transaction Management**: Handles MySQL transaction errors
- **Retry Logic**: Automatic retries for transient errors
- **Graceful Degradation**: Continues pipeline even if some steps fail
- **Detailed Logging**: Comprehensive error reporting

## Usage

### One-Click ETL Pipeline
```bash
python run_complete_etl_pipeline.py
```

### Step-by-Step ETL Pipeline
```bash
python src/etl/etl_pipeline_new.py
```

## Verification

After running the pipeline, verify:
1. All tables exist in MySQL database
2. All files uploaded to Azure storage
3. Row counts match expected values
4. Column counts are correct after dynamic cleaning

## Notes

1. **Backup**: All original data preserved in Azure
2. **Idempotent**: Can be run multiple times safely
3. **Scalable**: Designed for large datasets
4. **Auditable**: Complete logging of all operations 