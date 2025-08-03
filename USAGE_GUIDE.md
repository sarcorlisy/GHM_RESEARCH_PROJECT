# Usage Guide - Which File Should You Run

## Quick Start

Based on your needs, choose the corresponding file to run:

### 1. If you want to run the complete ETL data cleaning pipeline (Recommended)
```bash
python run_complete_etl_pipeline.py
```
**Function**: Download data from Azure → Preprocess and import to SQL → Mapping → Cleaning → Business cleaning → Upload back to Azure (one file completes all operations)

### 2. If you want to run dynamic column cleaning (Remove high invalid value columns)
```bash
python run_dynamic_cleaning.py
```
**Function**: Automatically identify and remove columns with invalid value ratios exceeding 50% (such as weight, max_glu_serum, A1Cresult)

### 3. If you want to run the complete data science pipeline
```bash
python main_pipeline.py
```
**Function**: Complete data loading, preprocessing, feature selection, and model training pipeline

### 4. If you want to run the example pipeline
```bash
python run_example.py
```
**Function**: Simplified example pipeline, suitable for quick testing

### 5. If you want to import data to MySQL
```bash
python import_data_to_mysql.py
```
**Function**: Import raw data to MySQL database

### 6. If you want to upload cleaned data to Azure
```bash
python upload_cleaned_tables.py
```
**Function**: Upload cleaned data tables to Azure storage

## File Function Description

### Core Execution Files
| Filename | Function | When to Use |
|----------|----------|-------------|
| `run_complete_etl_pipeline.py` | **Complete ETL Pipeline** | When you want to complete all data cleaning operations with one click |
| `run_dynamic_cleaning.py` | **Dynamic Column Cleaning** | When you want to remove high invalid value columns |
| `main_pipeline.py` | **Complete Data Science Pipeline** | When you want to run complete analysis |
| `run_example.py` | **Example Pipeline** | When you want to test quickly |
| `run_pipeline.py` | **Basic Pipeline** | When you want to run basic pipeline |

### Data Import/Upload Files
| Filename | Function | When to Use |
|----------|----------|-------------|
| `import_data_to_mysql.py` | **Import Data to MySQL** | When setting up database for the first time |
| `upload_cleaned_tables.py` | **Upload to Azure** | When you want to save cleaning results |
| `upload_mapped_data.py` | **Upload Mapped Data** | When you want to save mapping results |

### Analysis Files
| Filename | Function | When to Use |
|----------|----------|-------------|
| `sensitivity_analyzer.py` | **Sensitivity Analysis** | When you want to analyze different subgroups |
| `run_sensitivity_analysis.py` | **Run Sensitivity Analysis** | When you want to execute sensitivity analysis |
| `eda_analyzer.py` | **Exploratory Data Analysis** | When you want to analyze data distribution |

## Recommended Execution Order

### First Time Use (Complete Pipeline)
**Recommended: Complete all operations with one click**
```bash
python run_complete_etl_pipeline.py
```

**Or execute step by step:**
1. **Import Data**:
   ```bash
   python import_data_to_mysql.py
   ```

2. **Run Dynamic Column Cleaning**:
   ```bash
   python run_dynamic_cleaning.py
   ```

3. **Run Complete Analysis**:
   ```bash
   python main_pipeline.py
   ```

4. **Upload Results** (Optional):
   ```bash
   python upload_cleaned_tables.py
   ```

### Daily Use (Quick Cleaning)
If you only want to remove high invalid value columns:
```bash
python run_dynamic_cleaning.py
``` 