# Hospital Readmission Prediction - Data Pipeline Process

## Overview
This document describes the complete data pipeline process for the Hospital Readmission Prediction project, demonstrating enterprise-grade data engineering practices including Azure Data Lake integration, MySQL data processing, SQL cleaning, and feature engineering.

---

## Complete Data Pipeline

### Phase 1: Raw Data Preparation
**Files**: 
- `diabetic_data.csv` (Raw hospital data)
- `IDS_mapping.csv` (Mapping file)

**Upload to Azure**: 
- **Container**: `raw-data`
- **Paths**: 
  - `raw-data/diabetic_data.csv`
  - `raw-data/IDS_mapping.csv`

### Phase 2: Data Import and Preprocessing
**Python Script**: `import_data_to_mysql.py`

**Download from Azure**:
```python
# Priority download from Azure, fallback to local file
download_from_azure("raw-data/diabetic_data.csv")
```

**Enterprise-Grade 8-Step Data Cleaning**:
1. **Handle Missing Values**: `fillna('Unknown')`
2. **Handle Special Characters**: `replace('?', 'Unknown')`
3. **Ensure patient_id as String Type**
4. **Process Age Field**: Parse range format `[70-80)` → 75
5. **Process Numeric Fields**: Convert to numeric, fill NaN with 0, cast to int
6. **Standardize Medication Fields**: Convert to uppercase, 'UNKNOWN'/'NONE' → 'No'
7. **Clean Diagnosis Fields**: Strip whitespace, empty strings → 'Unknown'
8. **Data Quality Check**: Log total rows and null counts

**Business Rule Application**:
```python
# Keep only first admission record for each patient
df = df.sort_values('encounter_id').drop_duplicates('patient_nbr', keep='first')
```

**Result**: 71,518 rows imported into MySQL `patients` table

### Phase 3: Data Mapping Process
**Mapping File**: Download from Azure `raw-data/IDS_mapping.csv`

**Mapping Table Split** (following notebook logic):
```python
admission_type_df = ids_mapping_df.iloc[0:8]           # Admission type mapping
discharge_disposition_df = ids_mapping_df.iloc[10:40]  # Discharge disposition mapping  
admission_source_df = ids_mapping_df.iloc[42:]         # Admission source mapping
```

**Mapping Table Creation**:
```sql
-- Create 3 mapping tables
CREATE TABLE admission_type_mapping (admission_type_id INT, admission_type_desc VARCHAR(100));
CREATE TABLE discharge_disposition_mapping (discharge_disposition_id INT, discharge_disposition_desc VARCHAR(200));
CREATE TABLE admission_source_mapping (admission_source_id INT, admission_source_desc VARCHAR(200));
```

**Data Merging**:
```sql
-- Create mapped data table
CREATE TABLE patients_mapped AS
SELECT p.*, atm.admission_type_desc, ddm.discharge_disposition_desc, asm.admission_source_desc
FROM patients p
LEFT JOIN admission_type_mapping atm ON p.admission_type_id = atm.admission_type_id
LEFT JOIN discharge_disposition_mapping ddm ON p.discharge_disposition_id = ddm.discharge_disposition_id
LEFT JOIN admission_source_mapping asm ON p.admission_source_id = asm.admission_source_id;
```

**Result**: 71,518 rows, 53 columns (original 50 columns + 3 mapping description columns)

**Upload to Azure**: `processed-data/mapped_data.csv`

### Phase 4: Data Cleaning Process
**SQL Script**: `src/etl/sql_processing/04_data_cleaning.sql`

#### Step 1: Remove ID Columns + Invalid Value Analysis and Column Removal
```sql
-- Create patients_cleaned table, remove admission_type_id, discharge_disposition_id, admission_source_id
CREATE TABLE patients_cleaned AS
SELECT [all columns except 3 ID columns] FROM patients_mapped;

-- Analyze invalid value percentages and remove high invalid rate columns
-- Remove 3 high invalid rate columns: weight(96.01%), max_glu_serum(95.17%), A1Cresult(81.84%)
```

**Result**: 71,518 rows, 49 columns (removed 3 ID columns + 3 high invalid rate columns)

#### Step 2: Business Rule Filtering
```sql
-- Remove patients who cannot be readmitted
CREATE TABLE patients_business_cleaned AS
SELECT * FROM patients_final_cleaned
WHERE discharge_disposition_desc NOT IN (
    'Expired', 'Hospice / home', 'Hospice / medical facility',
    'Expired at home. Medicaid only, hospice.',
    'Expired in a medical facility. Medicaid only, hospice.',
    'Expired, place unknown. Medicaid only, hospice.'
);
```

**Result**: 69,973 rows, 49 columns (removed 1,545 records of patients who cannot be readmitted)

### Phase 5: Final Data Upload
**Python Script**: `upload_cleaned_tables.py`

**Upload Two Cleaned Tables to Azure**:
1. **`patients_cleaned.csv`**: 71,518 rows, 49 columns (removed ID columns + high invalid rate columns)
2. **`patients_business_cleaned.csv`**: 69,973 rows, 49 columns (final cleaned data)

---

## Data Pipeline Statistics

| Phase | Table Name | Rows | Columns | Description |
|-------|------------|------|---------|-------------|
| Raw | diabetic_data.csv | 101,766 | 50 | Raw data |
| Import | patients | 71,518 | 50 | First admission filter |
| Mapping | patients_mapped | 71,518 | 53 | +3 mapping description columns |
| Cleaning 1 | patients_cleaned | 71,518 | 49 | -3 ID columns + -3 high invalid rate columns |
| Cleaning 2 | patients_business_cleaned | 69,973 | 49 | -1,545 business filtered records |

---

## Technical Architecture

### Data Storage Layer
- **Azure Data Lake Storage**: Raw data, processing data, final data
- **MySQL Database**: Relational data storage and processing

### Processing Layer
- **Python**: Data import, cleaning, upload
- **SQL**: Data mapping, cleaning, business rules

### Container Structure
```
hospitalreadmission/
├── raw-data/
│   ├── diabetic_data.csv
│   └── IDS_mapping.csv
└── processed-data/
    ├── mapped_data.csv
    ├── patients_cleaned.csv
    └── patients_business_cleaned.csv
```

---

## Key Features

1. **Enterprise-Grade ETL Pipeline**: Complete process from raw data to cleaned data
2. **Azure Data Lake Integration**: Cloud storage and data processing
3. **SQL Data Cleaning**: Business rule-based data quality improvement
4. **Data Mapping**: Convert IDs to readable description information
5. **Business Rule Application**: Remove irrelevant patient records
6. **Data Quality Assurance**: Remove high invalid rate columns, improve data quality

**Final Result**: 69,973 rows of high-quality, cleaned hospital readmission prediction data ready for machine learning modeling. 