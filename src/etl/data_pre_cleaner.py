"""
Data Cleaning Module - 8 Major Cleaning Rules
Independent data cleaning functionality that can be called by other modules
"""

import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class DataPreCleaner:
    """Data Pre-cleaner - Implements 8 Major Cleaning Rules"""
    
    def __init__(self):
        """Initialize data cleaner"""
        pass
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute 8 major cleaning rules
        
        Args:
            df: Original data DataFrame
            
        Returns:
            df: Cleaned DataFrame
        """
        logger.info("Starting 8 major cleaning rules...")
        
        # 1. Handle missing values
        df = self._handle_missing_values(df)
        
        # 2. Handle special characters
        df = self._handle_special_characters(df)
        
        # 3. Standardize patient IDs
        df = self._standardize_patient_id(df)
        
        # 4. Process age fields
        df = self._process_age_field(df)
        
        # 5. Process numeric fields
        df = self._process_numeric_fields(df)
        
        # 6. Standardize medication fields
        df = self._standardize_medication_fields(df)
        
        # 7. Process diagnosis fields
        df = self._process_diagnosis_fields(df)
        
        # 8. Add timestamp
        df = self._add_timestamp(df)
        
        logger.info(f"8 major cleaning rules completed - Total rows: {len(df)}")
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rule 1: Handle missing values"""
        df = df.fillna('Unknown')
        logger.info("Rule 1: Missing value processing completed")
        return df
    
    def _handle_special_characters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rule 2: Handle special characters"""
        df = df.replace('?', 'Unknown')
        logger.info("Rule 2: Special character processing completed")
        return df
    
    def _standardize_patient_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rule 3: Standardize patient IDs"""
        if 'patient_nbr' in df.columns:
            df['patient_nbr'] = df['patient_nbr'].astype(str)
        logger.info("Rule 3: Patient ID standardization completed")
        return df
    
    def _process_age_field(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rule 4: Process age field - Extract midpoint of age range"""
        if 'age' in df.columns:
            def extract_age_midpoint(age_str):
                if pd.isna(age_str) or age_str == 'Unknown':
                    return None
                try:
                    if isinstance(age_str, str) and '[' in age_str and ')' in age_str:
                        age_range = age_str.replace('[', '').replace(')', '')
                        if '-' in age_range:
                            start, end = age_range.split('-')
                            return int((int(start) + int(end)) / 2)
                        else:
                            return int(age_range)
                    else:
                        return int(age_str)
                except:
                    return None
            
            df['age'] = df['age'].apply(extract_age_midpoint)
        logger.info("Rule 4: Age field processing completed")
        return df
    
    def _process_numeric_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rule 5: Process numeric fields"""
        numeric_columns = [
            'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
            'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
            'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0).astype(int)
        
        logger.info("Rule 5: Numeric field processing completed")
        return df
    
    def _standardize_medication_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rule 6: Standardize medication fields"""
        medication_columns = [
            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
            'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
            'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
            'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
            'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
        ]
        
        for col in medication_columns:
            if col in df.columns:
                df[col] = df[col].str.upper()
                df[col] = df[col].replace(['UNKNOWN', 'NONE'], 'No')
        
        logger.info("Rule 6: Medication field standardization completed")
        return df
    
    def _process_diagnosis_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rule 7: Process diagnosis fields"""
        diagnosis_columns = ['diag_1', 'diag_2', 'diag_3']
        for col in diagnosis_columns:
            if col in df.columns:
                df[col] = df[col].str.strip()
                df[col] = df[col].replace('', 'Unknown')
        
        logger.info("Rule 7: Diagnosis field processing completed")
        return df
    
    def _add_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rule 8: Add timestamp"""
        df['created_at'] = pd.Timestamp.now()
        logger.info("Rule 8: Timestamp addition completed")
        return df
    
    def apply_first_admission_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply first admission logic
        
        Args:
            df: Original data DataFrame
            
        Returns:
            df: DataFrame after first admission filtering
        """
        logger.info("Applying first admission logic...")
        
        # Sort by encounter_id, then keep the first admission record for each patient
        df_sorted = df.sort_values(by='encounter_id')
        df_first = df_sorted.drop_duplicates(subset='patient_nbr', keep='first')
        
        logger.info(f"First admission filtering completed: {len(df)} -> {len(df_first)} rows")
        return df_first 