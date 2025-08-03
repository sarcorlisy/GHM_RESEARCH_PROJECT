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
        
        # Remove rolling_avg in the final step of feature engineering to ensure subsequent df no longer contains this feature
        if 'rolling_avg' in df.columns:
            df = df.drop(columns=['rolling_avg'])
        
        return df
    
    def prepare_target_variable(self, df: pd.DataFrame, target_type: str = 'binary') -> pd.DataFrame:
        """
        Prepares the target variable with different encoding options
        
        Args:
            df: The input DataFrame
            target_type: Target variable type ('binary', 'multiclass', or 'risk_level')
            
        Returns:
            The DataFrame after processing the target variable
        """
        logger.info(f"Preparing target variable with {target_type} encoding...")
        
        if target_type == 'binary':
            # Binary classification: focus on <30 days readmission (most critical)
            # Both 'NO' and '>30' are considered safe (0), only '<30' is high risk (1)
            df['readmitted_binary'] = (df['readmitted'] == '<30').astype(int)
            logger.info("Binary target: 0=Safe (No readmission or >30 days), 1=High risk (<30 days readmission)")
            
        elif target_type == 'multiclass':
            # Multi-class classification with semantic encoding
            readmission_mapping = {
                'NO': 0,      # No readmission (baseline)
                '>30': 1,     # Readmitted after 30 days (less critical)
                '<30': 2      # Readmitted within 30 days (most critical)
            }
            df['readmitted_multiclass'] = df['readmitted'].map(readmission_mapping)
            logger.info("Multi-class target: 0=No readmission, 1=>30 days, 2=<30 days")
            
        elif target_type == 'risk_level':
            # Risk-based encoding (alternative approach)
            risk_mapping = {
                'NO': 0,      # No risk (no readmission)
                '>30': 1,     # Low risk (readmitted after 30 days)
                '<30': 3      # High risk (readmitted within 30 days)
            }
            df['readmitted_risk'] = df['readmitted'].map(risk_mapping)
            logger.info("Risk-based target: 0=No risk, 1=Low risk, 3=High risk")
            
        return df
    
    def split_data(self, df: pd.DataFrame, target_col: str = 'readmitted_binary',
                   test_size: float = 0.2, val_size: float = 0.2, 
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split the dataset and remove unused ID columns before splitting.

        Args:
            df: Input DataFrame
            target_col: Target variable column name
            test_size: Test set proportion
            val_size: Validation set proportion
            random_state: Random seed

        Returns:
            Features and target variables for train, validation, and test sets
        """
        logger.info("Splitting data into train/validation/test sets...")

        # 1. Define and remove ID columns that are not useful for model training
        cols_to_drop = [
            'encounter_id', 'patient_nbr', 
            'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
            'readmitted'  # Original target variable, replaced by binary version
        ]
        
        # Select columns that actually exist in the DataFrame for removal
        existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        if existing_cols_to_drop:
            logger.info(f"Dropping unused ID and target columns before splitting: {existing_cols_to_drop}")
            X = df.drop(columns=existing_cols_to_drop + [target_col])
        else:
            X = df.drop(columns=[target_col])

        y = df[target_col]

        # 2. First split out the test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # 3. Split out the validation set from the remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )

        logger.info(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def encode_categorical_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                                  X_test: pd.DataFrame, encoding_method: str = 'label') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Encode categorical features
        
        Args:
            X_train: Training set features
            X_val: Validation set features
            X_test: Test set features
            encoding_method: Encoding method ('label' or 'onehot')
            
        Returns:
            Encoded feature sets
        """
        logger.info(f"Encoding categorical features using {encoding_method} encoding...")
        
        categorical_features = X_train.select_dtypes(include=['object']).columns
        
        if encoding_method == 'label':
            # Label Encoding (suitable for tree models)
            for col in categorical_features:
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col])
                X_val[col] = X_val[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
                X_test[col] = X_test[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
                self.label_encoders[col] = le
                
        elif encoding_method == 'onehot':
            # One-Hot Encoding (suitable for linear models)
            from sklearn.preprocessing import OneHotEncoder
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            
            # Perform One-Hot encoding on training set
            train_encoded = ohe.fit_transform(X_train[categorical_features])
            val_encoded = ohe.transform(X_val[categorical_features])
            test_encoded = ohe.transform(X_test[categorical_features])
            
            # Create new column names
            feature_names = ohe.get_feature_names_out(categorical_features)
            
            # Delete original categorical columns
            X_train = X_train.drop(columns=categorical_features)
            X_val = X_val.drop(columns=categorical_features)
            X_test = X_test.drop(columns=categorical_features)
            
            # Add encoded columns
            X_train = pd.concat([X_train, pd.DataFrame(train_encoded, columns=feature_names, index=X_train.index)], axis=1)
            X_val = pd.concat([X_val, pd.DataFrame(val_encoded, columns=feature_names, index=X_val.index)], axis=1)
            X_test = pd.concat([X_test, pd.DataFrame(test_encoded, columns=feature_names, index=X_test.index)], axis=1)
            
            self.onehot_encoder = ohe
        
        return X_train, X_val, X_test
    
    def scale_numerical_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                                X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Standardize numerical features
        
        Args:
            X_train: Training set features
            X_val: Validation set features
            X_test: Test set features
            
        Returns:
            Standardized feature sets
        """
        logger.info("Scaling numerical features...")
        
        numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
        
        # Fit the scaler
        self.scaler.fit(X_train[numerical_features])
        
        # Transform all datasets
        X_train[numerical_features] = self.scaler.transform(X_train[numerical_features])
        X_val[numerical_features] = self.scaler.transform(X_val[numerical_features])
        X_test[numerical_features] = self.scaler.transform(X_test[numerical_features])
        
        return X_train, X_val, X_test
    
    def apply_smote(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE to balance the training dataset.

        Args:
            X_train: Training set features
            y_train: Training set labels
            random_state: Random seed

        Returns:
            Balanced training set features and labels
        """
        logger.info("Applying SMOTE for class balancing...")
        
        smote = SMOTE(random_state=random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        logger.info(f"Before SMOTE - Class distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"After SMOTE - Class distribution: {y_train_balanced.value_counts().to_dict()}")
        
        return X_train_balanced, y_train_balanced
    
    def get_feature_categories(self) -> Dict[str, List[str]]:
        """
        Get feature category information
        
        Returns:
            Feature category dictionary
        """
        return FEATURE_CATEGORIES
    
    def save_preprocessed_data(self, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                             X_test: pd.DataFrame, y_train: pd.Series, y_val: pd.Series, 
                             y_test: pd.Series, output_dir: str = 'outputs') -> None:
        """
        Save preprocessed data
        
        Args:
            X_train: Training set features
            X_val: Validation set features
            X_test: Test set features
            y_train: Training set target variable
            y_val: Validation set target variable
            y_test: Test set target variable
            output_dir: Output directory
        """
        logger.info("Saving preprocessed data...")
        
        # Create output directory
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save data
        X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
        X_val.to_csv(f"{output_dir}/X_val.csv", index=False)
        X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
        y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
        y_val.to_csv(f"{output_dir}/y_val.csv", index=False)
        y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
        
        logger.info(f"Preprocessed data saved to {output_dir}")

def main():
    """Main function for testing data preprocessing functionality"""
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    df = loader.merge_data()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Apply feature engineering
    df = preprocessor.apply_feature_engineering(df)
    
    # Prepare target variable
    df = preprocessor.prepare_target_variable(df)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(df)
    
    # Encode categorical features
    X_train, X_val, X_test = preprocessor.encode_categorical_features(X_train, X_val, X_test)
    
    # Standardize numerical features
    X_train, X_val, X_test = preprocessor.scale_numerical_features(X_train, X_val, X_test)
    
    # Apply SMOTE
    X_train_balanced, y_train_balanced = preprocessor.apply_smote(X_train, y_train)
    
    # Save preprocessed data
    preprocessor.save_preprocessed_data(X_train_balanced, X_val, X_test, 
                                      y_train_balanced, y_val, y_test)
    
    print("Data preprocessing completed successfully!")
    return X_train_balanced, X_val, X_test, y_train_balanced, y_val, y_test

if __name__ == "__main__":
    main() 