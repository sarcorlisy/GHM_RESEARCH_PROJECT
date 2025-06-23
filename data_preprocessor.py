"""
Data Preprocessing Module for Hospital Readmission Prediction Pipeline
"""
import pandas as pd

import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import logging
from typing import Dict, Tuple, List, Optional
import warnings

from pipeline_config import ICD9_CATEGORIES, FEATURE_CATEGORIES

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

class DataPreprocessor:
    """Data preprocessor class, responsible for feature engineering and data cleaning"""
    
    def __init__(self):
        """Initializes the preprocessor"""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_engineering_applied = False
        
    def icd9_to_nine_category(self, code: str) -> str:
        """
        Classifies ICD-9 codes into 9 major categories
        
        Args:
            code: The ICD-9 code
            
        Returns:
            The classification result
        """
        try:
            code = str(code).strip()
            if code.startswith('E') or code.startswith('V'):
                return 'other'
            
            num = float(code)
            
            for category, (start, end) in ICD9_CATEGORIES.items():
                if start <= num <= end:
                    return category
            
            return 'other'
        except:
            return 'other'
    
    def create_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates age-related features
        
        Args:
            df: The input DataFrame
            
        Returns:
            The DataFrame with added age features
        """
        logger.info("Creating age-related features...")
        
        # Create age midpoint
        df['age_midpoint'] = df['age'].apply(lambda x: 
            int(x.replace('[', '').replace(')', '').split('-')[0]) + 
            int(x.replace('[', '').replace(')', '').split('-')[1]) // 2
        )
        
        # Create age group
        df['age_group'] = df['age'].apply(lambda x: 
            x.replace('[', '').replace(')', '')
        )
        
        return df
    
    def create_diagnosis_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates diagnosis-related features
        
        Args:
            df: The input DataFrame
            
        Returns:
            The DataFrame with added diagnosis features
        """
        logger.info("Creating diagnosis-related features...")
        
        # Create categorical features for each diagnosis
        for col in ['diag_1', 'diag_2', 'diag_3']:
            df[f"{col}_category"] = df[col].apply(self.icd9_to_nine_category)
        
        return df
    
    def create_comorbidity_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates the comorbidity feature
        
        Args:
            df: The input DataFrame
            
        Returns:
            The DataFrame with the added comorbidity feature
        """
        logger.info("Creating comorbidity feature...")
        
        # Calculate comorbidity count (based on number of diagnoses)
        df['comorbidity'] = df['number_diagnoses'].apply(lambda x: 
            0 if x <= 1 else (1 if x <= 3 else 2)
        )
        
        return df
    
    def create_encounter_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates encounter-related features
        
        Args:
            df: The input DataFrame
            
        Returns:
            The DataFrame with added encounter features
        """
        logger.info("Creating encounter-related features...")
        
        # Create encounter index
        df['encounter_index'] = df.groupby('patient_nbr').cumcount() + 1
        
        # Create rolling average feature
        df['rolling_avg'] = df.groupby('patient_nbr')['time_in_hospital'].transform(
            lambda x: x.expanding().mean()
        )
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, drop_missing_threshold: float = 50.0) -> pd.DataFrame:
        """
        Handles missing values according to the original notebook's standard.

        - First, replaces all '?' with NaN.
        - Drops columns with a missing rate exceeding `drop_missing_threshold`%.
        - Fills remaining missing values in specified columns with 'Unknown'.

        Args:
            df: The input DataFrame.
            drop_missing_threshold: The missing rate threshold for dropping columns (in percent).

        Returns:
            The DataFrame after handling missing values.
        """
        logger.info("Handling missing values based on notebook's standard...")

        # 1. Replace '?' with NaN
        df = df.replace('?', np.nan)
        logger.info("Replaced '?' with NaN.")

        # 2. Calculate and drop columns with high missing rates
        missing_percentage = (df.isnull().sum() / len(df)) * 100
        cols_to_drop = missing_percentage[missing_percentage > drop_missing_threshold].index
        if len(cols_to_drop) > 0:
            df = df.drop(columns=cols_to_drop)
            logger.info(f"🔴 Dropped columns with >{drop_missing_threshold}% missing: {list(cols_to_drop)}")
        else:
            logger.info(f"✅ No columns with >{drop_missing_threshold}% missing values to drop.")

        # 3. Fill remaining missing values
        cols_to_fill = [
            'medical_specialty', 'payer_code', 'race',
            'diag_1', 'diag_2', 'diag_3',
            'admission_type_desc', 'discharge_disposition_desc', 'admission_source_desc'
        ]
        
        # Filter for columns that actually exist in the DataFrame and need filling
        existing_cols_to_fill = [col for col in cols_to_fill if col in df.columns and df[col].isnull().any()]

        if existing_cols_to_fill:
            logger.info(f"🟡 Filling specified columns with 'Unknown': {existing_cols_to_fill}")
            df[existing_cols_to_fill] = df[existing_cols_to_fill].fillna('Unknown')
        else:
            logger.info("✅ No missing values found in the specified columns to fill.")

        # Final check
        final_missing = df.isnull().sum().sum()
        if final_missing == 0:
            logger.info("✅ All missing values have been handled successfully.")
        else:
            logger.warning(f"⚠️ Still have {final_missing} missing values after processing.")
            remaining_cols = df.isnull().sum()
            logger.warning(f"Columns with remaining NaNs:\n{remaining_cols[remaining_cols > 0]}")

        return df
    
    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies feature engineering
        
        Args:
            df: The input DataFrame
            
        Returns:
            The DataFrame after feature engineering
        """
        logger.info("Applying feature engineering...")
        
        # 1. Keep only the first admission record for each patient
        logger.info(f"Original number of encounters: {len(df)}")
        df_sorted = df.sort_values(by='encounter_id')
        df = df_sorted.drop_duplicates(subset='patient_nbr', keep='first')
        logger.info(f"Encounters after keeping first admission: {len(df)}")
        
        # 2. Remove patients who died or went to hospice
        hospice_or_death_ids = [11, 13, 14, 19, 20, 21]
        records_before_filter = len(df)
        df = df[~df['discharge_disposition_id'].isin(hospice_or_death_ids)]
        logger.info(f"Removed {records_before_filter - len(df)} records for hospice/death dispositions.")
        logger.info(f"Encounters after removing hospice/death: {len(df)}")

        # 3. Handle missing values
        df = self.handle_missing_values(df)
        
        # 4. Create various features
        df = self.create_age_features(df)
        df = self.create_diagnosis_features(df)
        df = self.create_comorbidity_feature(df)
        df = self.create_encounter_features(df)
        
        self.feature_engineering_applied = True
        logger.info("Feature engineering completed")
        
        return df
    
    def prepare_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares the target variable
        
        Args:
            df: The input DataFrame
            
        Returns:
            The DataFrame after processing the target variable
        """
        logger.info("Preparing target variable...")
        
        # Convert the target variable to binary (0: not readmitted, 1: readmitted)
        df['readmitted_binary'] = (df['readmitted'] == '<30').astype(int)
        
        return df
    
    def split_data(self, df: pd.DataFrame, target_col: str = 'readmitted_binary',
                   test_size: float = 0.2, val_size: float = 0.2, 
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        分割数据集，并在分割前删除无用的ID列。

        Args:
            df: 输入数据框
            target_col: 目标变量列名
            test_size: 测试集比例
            val_size: 验证集比例
            random_state: 随机种子

        Returns:
            训练集、验证集、测试集的特征和目标变量
        """
        logger.info("Splitting data into train/validation/test sets...")

        # 1. 定义并删除对模型训练无用的ID列
        cols_to_drop = [
            'encounter_id', 'patient_nbr', 
            'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
            'readmitted'  # 原始目标变量，已被二进制版本替代
        ]
        
        # 筛选出实际存在于DataFrame中的列进行删除
        existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        if existing_cols_to_drop:
            logger.info(f"Dropping unused ID and target columns before splitting: {existing_cols_to_drop}")
            X = df.drop(columns=existing_cols_to_drop + [target_col])
        else:
            X = df.drop(columns=[target_col])

        y = df[target_col]

        # 2. 首先分割出测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # 3. 从剩余数据中分割出验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )

        logger.info(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def encode_categorical_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                                  X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        编码分类特征
        
        Args:
            X_train: 训练集特征
            X_val: 验证集特征
            X_test: 测试集特征
            
        Returns:
            编码后的特征集
        """
        logger.info("Encoding categorical features...")
        
        categorical_features = X_train.select_dtypes(include=['object']).columns
        
        for col in categorical_features:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col])
            X_val[col] = X_val[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
            X_test[col] = X_test[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
            self.label_encoders[col] = le
        
        return X_train, X_val, X_test
    
    def scale_numerical_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                                X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        标准化数值特征
        
        Args:
            X_train: 训练集特征
            X_val: 验证集特征
            X_test: 测试集特征
            
        Returns:
            标准化后的特征集
        """
        logger.info("Scaling numerical features...")
        
        numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
        
        # 拟合标准化器
        self.scaler.fit(X_train[numerical_features])
        
        # 转换所有数据集
        X_train[numerical_features] = self.scaler.transform(X_train[numerical_features])
        X_val[numerical_features] = self.scaler.transform(X_val[numerical_features])
        X_test[numerical_features] = self.scaler.transform(X_test[numerical_features])
        
        return X_train, X_val, X_test
    
    def apply_smote(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
        """
        应用SMOTE来平衡训练数据集。

        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            random_state: 随机种子

        Returns:
            平衡后的训练集特征和标签
        """
        logger.info("Applying SMOTE for class balancing...")
        
        smote = SMOTE(random_state=random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        logger.info(f"Before SMOTE - Class distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"After SMOTE - Class distribution: {y_train_balanced.value_counts().to_dict()}")
        
        return X_train_balanced, y_train_balanced
    
    def get_feature_categories(self) -> Dict[str, List[str]]:
        """
        获取特征分类信息
        
        Returns:
            特征分类字典
        """
        return FEATURE_CATEGORIES
    
    def save_preprocessed_data(self, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                             X_test: pd.DataFrame, y_train: pd.Series, y_val: pd.Series, 
                             y_test: pd.Series, output_dir: str = 'outputs') -> None:
        """
        保存预处理后的数据
        
        Args:
            X_train: 训练集特征
            X_val: 验证集特征
            X_test: 测试集特征
            y_train: 训练集目标变量
            y_val: 验证集目标变量
            y_test: 测试集目标变量
            output_dir: 输出目录
        """
        logger.info("Saving preprocessed data...")
        
        # 创建输出目录
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存数据
        X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
        X_val.to_csv(f"{output_dir}/X_val.csv", index=False)
        X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
        y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
        y_val.to_csv(f"{output_dir}/y_val.csv", index=False)
        y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
        
        logger.info(f"Preprocessed data saved to {output_dir}")

def main():
    """主函数，用于测试数据预处理功能"""
    from data_loader import DataLoader
    
    # 加载数据
    loader = DataLoader()
    df = loader.merge_data()
    
    # 初始化预处理器
    preprocessor = DataPreprocessor()
    
    # 应用特征工程
    df = preprocessor.apply_feature_engineering(df)
    
    # 准备目标变量
    df = preprocessor.prepare_target_variable(df)
    
    # 分割数据
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(df)
    
    # 编码分类特征
    X_train, X_val, X_test = preprocessor.encode_categorical_features(X_train, X_val, X_test)
    
    # 标准化数值特征
    X_train, X_val, X_test = preprocessor.scale_numerical_features(X_train, X_val, X_test)
    
    # 应用SMOTE
    X_train_balanced, y_train_balanced = preprocessor.apply_smote(X_train, y_train)
    
    # 保存预处理后的数据
    preprocessor.save_preprocessed_data(X_train_balanced, X_val, X_test, 
                                      y_train_balanced, y_val, y_test)
    
    print("Data preprocessing completed successfully!")
    return X_train_balanced, X_val, X_test, y_train_balanced, y_val, y_test

if __name__ == "__main__":
    main() 