# Git Branch Strategy

## Overview
This project uses a dual-branch strategy to maintain both the original notebook-based approach and the new enterprise-grade SQL-Azure pipeline approach.

## Branch Structure

### `main` Branch
**Purpose**: Original notebook-based hospital readmission prediction project
**Content**:
- Original Jupyter notebooks for data analysis
- Python scripts for ML modeling
- Original project structure and documentation
- Sensitivity analysis notebooks
- Feature selection and model training scripts

**Key Files**:
- `Demo0720 main.ipynb` - Main analysis notebook
- `model_trainer.py` - ML model training
- `feature_selector.py` - Feature selection
- `sensitivity_analysis_demo.ipynb` - Sensitivity analysis
- Various analysis notebooks and results

### `sql-azure-pipeline` Branch
**Purpose**: Enterprise-grade SQL-Azure data pipeline implementation
**Content**:
- SQL-based data processing and cleaning
- Azure Data Lake integration
- MySQL database operations
- ETL pipeline implementation
- Enterprise-grade project structure

**Key Files**:
- `import_data_to_mysql.py` - Data import from Azure to MySQL
- `upload_mapped_data.py` - Upload mapped data to Azure
- `upload_cleaned_tables.py` - Upload cleaned data to Azure
- `src/etl/sql_processing/` - SQL processing scripts
- `config/` - Configuration files
- `docs/DATA_PIPELINE_PROCESS.md` - Pipeline documentation

## Branch Comparison

| Aspect | `main` Branch | `sql-azure-pipeline` Branch |
|--------|---------------|------------------------------|
| **Data Processing** | Pandas-based in notebooks | SQL-based with MySQL |
| **Storage** | Local CSV files | Azure Data Lake Storage |
| **Architecture** | Notebook-centric | Enterprise ETL pipeline |
| **Data Flow** | In-memory processing | Database + Cloud storage |
| **Use Case** | Research/Prototyping | Production/Enterprise |
| **Skills Demonstrated** | Data Science, ML | Data Engineering, ETL |

## Usage Instructions

### Working with `main` Branch
```bash
git checkout main
# Work with original notebook-based approach
```

### Working with `sql-azure-pipeline` Branch
```bash
git checkout sql-azure-pipeline
# Work with enterprise SQL-Azure pipeline
```

## Development Workflow

1. **Feature Development**: Create feature branches from appropriate base branch
2. **Testing**: Test changes in feature branch
3. **Merge**: Merge feature branch back to appropriate main branch
4. **Documentation**: Update relevant documentation

## Branch Naming Convention

- `main` - Original project
- `sql-azure-pipeline` - Enterprise pipeline implementation
- `feature/*` - Feature development branches
- `hotfix/*` - Critical bug fixes

## Commit Messages

### `main` Branch
- `feat: Add new ML model`
- `fix: Correct feature selection logic`
- `docs: Update notebook documentation`

### `sql-azure-pipeline` Branch
- `feat: Add SQL data cleaning step`
- `fix: Resolve Azure connection issue`
- `docs: Update pipeline documentation`

## Benefits of This Strategy

1. **Preserve Original Work**: Keep original notebook-based approach intact
2. **Demonstrate Growth**: Show progression from research to enterprise
3. **Flexible Presentation**: Choose appropriate branch for different audiences
4. **Version Control**: Track evolution of project approach
5. **Skill Demonstration**: Show both data science and data engineering skills

## Migration Path

The `sql-azure-pipeline` branch represents the evolution from:
- **Research Phase** (`main`) → **Enterprise Phase** (`sql-azure-pipeline`)
- **Notebook-based** → **Production-ready ETL pipeline**
- **Local Processing** → **Cloud-based Data Lake architecture** 