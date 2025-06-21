"""
Hospital Readmission Prediction Pipeline Configuration
"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 数据文件路径
DATA_PATHS = {
    'diabetic_data': PROJECT_ROOT / 'diabetic_data.csv',
    'ids_mapping': PROJECT_ROOT / 'IDS_mapping.csv',
    'ccs_mapping': PROJECT_ROOT / 'ccs_icd9_mapping.csv',
    'output_dir': PROJECT_ROOT / 'outputs'
}

# 创建输出目录
DATA_PATHS['output_dir'].mkdir(exist_ok=True)

# 模型配置
MODEL_CONFIG = {
    'test_size': 0.2,
    'val_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'feature_selection_top_n': 15
}

# 特征选择方法
FEATURE_SELECTION_METHODS = {
    'L1': 'Logistic Regression with L1 penalty',
    'MutualInfo': 'Mutual Information',
    'TreeImportance': 'Random Forest Feature Importance'
}

# 模型类型
MODELS = {
    'LogisticRegression': 'Logistic Regression',
    'RandomForest': 'Random Forest',
    'XGBoost': 'XGBoost'
}

# 特征分类定义
FEATURE_CATEGORIES = {
    "Demographic": [
        'race', 'gender', 'age', 'age_midpoint', 'age_group'
    ],
    "Administrative": [
        'encounter_index', 'admission_type_desc', 'admission_source_desc',
        'discharge_disposition_desc', 'payer_code', 'change'
    ],
    "Clinical": [
        'diag_1', 'diag_2', 'diag_3',
        'number_diagnoses', 'number_inpatient', 'number_outpatient', 'number_emergency',
        'medical_specialty', 'comorbidity', 'diag_1_category', 'diag_2_category', 'diag_3_category'
    ],
    "Utilization": [
        'num_lab_procedures', 'num_procedures', 'num_medications',
        'time_in_hospital', 'rolling_avg'
    ],
    "Medication": [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide',
        'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone',
        'diabetesMed'
    ],
    "Label": ['readmitted']
}

# ICD-9 分类映射
ICD9_CATEGORIES = {
    'circulatory': (390, 459),
    'respiratory': (460, 519),
    'digestive': (520, 579),
    'diabetes': (250.0, 251.0),
    'injuries': (800, 999),
    'musculoskeletal': (710, 739),
    'genitourinary': (580, 629),
    'neoplasms': (140, 239)
}

# 可视化配置
PLOT_CONFIG = {
    'figsize': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'save_format': 'png'
} 