# Hospital Readmission Prediction Pipeline - Structure Documentation

## Overall Architecture

```
Hospital Readmission Pipeline
├── Configuration Layer
│   └── pipeline_config.py
├── Data Layer
│   └── data_loader.py
├── Preprocessing Layer
│   └── data_preprocessor.py
├── Feature Engineering Layer
│   └── feature_selector.py
├── Model Layer
│   └── model_trainer.py
├── Control Layer
│   └── main_pipeline.py
└── Utility Layer
    ├── run_example.py
    ├── test_pipeline.py
    └── requirements.txt
```

## Module Detailed Description

### 1. Configuration Layer (pipeline_config.py)

**Responsibility**: Centralized management of all configuration parameters

**Main Components**:
- `DATA_PATHS`: Data file path configuration
- `MODEL_CONFIG`: Model training parameter configuration
- `FEATURE_SELECTION_METHODS`: Feature selection method definitions
- `MODELS`: Available model type definitions
- `FEATURE_CATEGORIES`: Feature category definitions
- `ICD9_CATEGORIES`: ICD-9 code classification mappings
- `PLOT_CONFIG`: Visualization configuration

**Advantages**:
- Centralized configuration management
- Easy to modify and maintain
- Support for different environment configurations

### 2. Data Layer (data_loader.py)

**Responsibility**: Data loading and merging

**Main Functions**:
- `DataLoader.load_diabetic_data()`: Load diabetes dataset
- `DataLoader.load_ids_mapping()`: Load ID mapping data
- `DataLoader.split_ids_mapping()`: Split ID mapping tables
- `DataLoader.merge_data()`: Merge all data tables
- `DataLoader.get_data_info()`: Get data summary information
- `DataLoader.save_merged_data()`: Save merged data

**Data Flow**:
```
Raw data files → Load → Split mapping tables → Merge → Output merged data
```

### 3. Preprocessing Layer (data_preprocessor.py)

**Responsibility**: Data cleaning, feature engineering, and data transformation

**Main Functions**:

#### Feature Engineering
- `create_age_features()`: Create age-related features
- `create_diagnosis_features()`: Create diagnosis classification features
- `create_comorbidity_feature()`: Create comorbidity features
- `create_encounter_features()`: Create encounter-related features

#### Data Cleaning
- `handle_missing_values()`: Handle missing values
- Handle special characters '?'

#### Data Transformation
- `encode_categorical_features()`: Categorical feature encoding
- `scale_numerical_features()`: Numerical feature standardization
- `prepare_target_variable()`: Target variable preparation

#### Data Balancing
- `apply_smote()`: Use SMOTE to handle class imbalance

**Data Flow**:
```
Merged data → Feature engineering → Data cleaning → Data transformation → Data splitting → Data balancing → Output preprocessed data
```

### 4. Feature Engineering Layer (feature_selector.py)

**Responsibility**: Feature selection and importance analysis

**Main Functions**:
- `select_features_by_l1()`: L1 regularization feature selection
- `select_features_by_mi()`: Mutual information feature selection
- `select_features_by_tree()`: Tree model feature importance selection
- `select_all_features()`: Use all methods to select features
- `get_common_features()`: Get commonly selected features 