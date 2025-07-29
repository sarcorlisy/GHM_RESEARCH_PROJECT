"""
Data Migration Module for Hospital Readmission Prediction Pipeline

This module handles the migration of CSV data to PostgreSQL database,
including data validation, transformation, and bulk loading operations.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
import os
from datetime import datetime
import json
from database.db_connector import DatabaseConnector, DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """
    Data validation class for ensuring data quality during migration.
    
    This class provides methods for:
    - Data type validation
    - Missing value analysis
    - Data range validation
    - Consistency checks
    """
    
    def __init__(self):
        """Initialize data validator."""
        self.validation_results = {}
    
    def validate_patient_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate patient data for migration.
        
        Args:
            df: Patient DataFrame to validate
            
        Returns:
            Dict: Validation results
        """
        validation_results = {
            'total_rows': len(df),
            'missing_values': {},
            'data_types': {},
            'range_checks': {},
            'consistency_checks': {},
            'is_valid': True
        }
        
        # Check missing values
        missing_counts = df.isnull().sum()
        validation_results['missing_values'] = missing_counts.to_dict()
        
        # Check data types
        validation_results['data_types'] = df.dtypes.to_dict()
        
        # Range checks for numeric columns
        if 'age' in df.columns:
            validation_results['range_checks']['age'] = {
                'min': df['age'].min(),
                'max': df['age'].max(),
                'valid_range': (0, 120)
            }
        
        if 'time_in_hospital' in df.columns:
            validation_results['range_checks']['time_in_hospital'] = {
                'min': df['time_in_hospital'].min(),
                'max': df['time_in_hospital'].max(),
                'valid_range': (1, 365)
            }
        
        # Consistency checks
        if 'readmitted' in df.columns:
            valid_readmission_values = ['<30', '>30', 'NO']
            invalid_values = df[~df['readmitted'].isin(valid_readmission_values)]['readmitted'].unique()
            validation_results['consistency_checks']['readmitted'] = {
                'valid_values': valid_readmission_values,
                'invalid_values': invalid_values.tolist(),
                'is_consistent': len(invalid_values) == 0
            }
        
        # Overall validation
        validation_results['is_valid'] = (
            validation_results['missing_values'].get('patient_id', 0) == 0 and
            validation_results['consistency_checks'].get('readmitted', {}).get('is_consistent', False)
        )
        
        return validation_results
    
    def validate_medication_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate medication data for migration.
        
        Args:
            df: Medication DataFrame to validate
            
        Returns:
            Dict: Validation results
        """
        validation_results = {
            'total_rows': len(df),
            'missing_values': {},
            'data_types': {},
            'consistency_checks': {},
            'is_valid': True
        }
        
        # Check missing values
        missing_counts = df.isnull().sum()
        validation_results['missing_values'] = missing_counts.to_dict()
        
        # Check data types
        validation_results['data_types'] = df.dtypes.to_dict()
        
        # Consistency checks for medication columns
        if 'medication_name' in df.columns:
            unique_medications = df['medication_name'].nunique()
            validation_results['consistency_checks']['medication_name'] = {
                'unique_count': unique_medications,
                'is_consistent': unique_medications > 0
            }
        
        # Overall validation
        validation_results['is_valid'] = (
            validation_results['missing_values'].get('patient_id', 0) == 0
        )
        
        return validation_results


class DataTransformer:
    """
    Data transformation class for preparing data for database migration.
    
    This class provides methods for:
    - Data type conversion
    - Missing value handling
    - Data cleaning
    - Format standardization
    """
    
    def __init__(self):
        """Initialize data transformer."""
        self.transformation_log = []
    
    def transform_patient_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform patient data for database migration.
        
        Args:
            df: Raw patient DataFrame
            
        Returns:
            pd.DataFrame: Transformed DataFrame
        """
        df_transformed = df.copy()
        
        # Handle missing values
        df_transformed = self._handle_missing_values(df_transformed)
        
        # Convert data types
        df_transformed = self._convert_data_types(df_transformed)
        
        # Clean special characters
        df_transformed = self._clean_special_characters(df_transformed)
        
        # Add timestamp
        df_transformed['created_at'] = datetime.now()
        
        self.transformation_log.append({
            'operation': 'transform_patient_data',
            'original_rows': len(df),
            'transformed_rows': len(df_transformed),
            'timestamp': datetime.now()
        })
        
        return df_transformed
    
    def transform_medication_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform medication data for database migration.
        
        Args:
            df: Raw medication DataFrame
            
        Returns:
            pd.DataFrame: Transformed DataFrame
        """
        df_transformed = df.copy()
        
        # Handle missing values
        df_transformed = self._handle_missing_values(df_transformed)
        
        # Clean medication names
        if 'medication_name' in df_transformed.columns:
            df_transformed['medication_name'] = df_transformed['medication_name'].str.strip()
        
        # Add timestamp
        df_transformed['created_at'] = datetime.now()
        
        self.transformation_log.append({
            'operation': 'transform_medication_data',
            'original_rows': len(df),
            'transformed_rows': len(df_transformed),
            'timestamp': datetime.now()
        })
        
        return df_transformed
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in DataFrame."""
        # Replace '?' with None for string columns
        string_columns = df.select_dtypes(include=['object']).columns
        for col in string_columns:
            df[col] = df[col].replace('?', None)
        
        # Fill numeric missing values with appropriate defaults
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in ['age', 'time_in_hospital', 'num_medications', 'num_lab_procedures']:
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert data types for database compatibility."""
        # Convert age to integer
        if 'age' in df.columns:
            df['age'] = pd.to_numeric(df['age'], errors='coerce').astype('Int64')
        
        # Convert time_in_hospital to integer
        if 'time_in_hospital' in df.columns:
            df['time_in_hospital'] = pd.to_numeric(df['time_in_hospital'], errors='coerce').astype('Int64')
        
        # Convert medication counts to integer
        medication_count_columns = ['num_medications', 'num_lab_procedures', 'num_procedures']
        for col in medication_count_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        
        return df
    
    def _clean_special_characters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean special characters from string columns."""
        string_columns = df.select_dtypes(include=['object']).columns
        for col in string_columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
        
        return df


class DataMigrator:
    """
    Main data migration class for handling CSV to database migration.
    
    This class orchestrates the entire migration process including:
    - Data validation
    - Data transformation
    - Database loading
    - Migration reporting
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize data migrator.
        
        Args:
            db_manager: DatabaseManager instance
        """
        self.db_manager = db_manager
        self.validator = DataValidator()
        self.transformer = DataTransformer()
        self.migration_log = []
    
    def migrate_csv_to_database(self, csv_file_path: str, table_name: str, 
                               validate_only: bool = False) -> Dict[str, Any]:
        """
        Migrate CSV file to database table.
        
        Args:
            csv_file_path: Path to CSV file
            table_name: Target table name
            validate_only: If True, only validate without migrating
            
        Returns:
            Dict: Migration results
        """
        migration_result = {
            'file_path': csv_file_path,
            'table_name': table_name,
            'validation_results': None,
            'transformation_results': None,
            'migration_success': False,
            'error_message': None,
            'timestamp': datetime.now()
        }
        
        try:
            # Load CSV data
            logger.info(f"Loading CSV file: {csv_file_path}")
            df = pd.read_csv(csv_file_path)
            
            # Validate data
            logger.info("Validating data...")
            if table_name == 'patients':
                validation_results = self.validator.validate_patient_data(df)
            elif table_name == 'medications':
                validation_results = self.validator.validate_medication_data(df)
            else:
                validation_results = {'is_valid': True, 'total_rows': len(df)}
            
            migration_result['validation_results'] = validation_results
            
            if not validation_results.get('is_valid', True):
                migration_result['error_message'] = "Data validation failed"
                return migration_result
            
            if validate_only:
                migration_result['migration_success'] = True
                return migration_result
            
            # Transform data
            logger.info("Transforming data...")
            if table_name == 'patients':
                df_transformed = self.transformer.transform_patient_data(df)
            elif table_name == 'medications':
                df_transformed = self.transformer.transform_medication_data(df)
            else:
                df_transformed = df
            
            migration_result['transformation_results'] = {
                'original_rows': len(df),
                'transformed_rows': len(df_transformed),
                'transformation_log': self.transformer.transformation_log
            }
            
            # Migrate to database
            logger.info(f"Migrating data to table: {table_name}")
            migration_success = self.db_manager.migrate_csv_to_database(
                df_transformed, table_name
            )
            
            migration_result['migration_success'] = migration_success
            
            if migration_success:
                logger.info(f"Successfully migrated {len(df_transformed)} rows to {table_name}")
                self.migration_log.append(migration_result)
            else:
                migration_result['error_message'] = "Database migration failed"
            
        except Exception as e:
            logger.error(f"Migration failed: {str(e)}")
            migration_result['error_message'] = str(e)
        
        return migration_result
    
    def migrate_all_data(self, data_files: Dict[str, str], 
                        validate_only: bool = False) -> Dict[str, Any]:
        """
        Migrate all data files to database.
        
        Args:
            data_files: Dictionary mapping table names to CSV file paths
            validate_only: If True, only validate without migrating
            
        Returns:
            Dict: Overall migration results
        """
        overall_results = {
            'total_files': len(data_files),
            'successful_migrations': 0,
            'failed_migrations': 0,
            'migration_details': {},
            'overall_success': False,
            'timestamp': datetime.now()
        }
        
        for table_name, file_path in data_files.items():
            logger.info(f"Starting migration for {table_name}")
            
            migration_result = self.migrate_csv_to_database(
                file_path, table_name, validate_only
            )
            
            overall_results['migration_details'][table_name] = migration_result
            
            if migration_result['migration_success']:
                overall_results['successful_migrations'] += 1
            else:
                overall_results['failed_migrations'] += 1
        
        overall_results['overall_success'] = (
            overall_results['failed_migrations'] == 0
        )
        
        return overall_results
    
    def generate_migration_report(self, migration_results: Dict[str, Any]) -> str:
        """
        Generate detailed migration report.
        
        Args:
            migration_results: Migration results dictionary
            
        Returns:
            str: Formatted migration report
        """
        report_lines = [
            "=" * 80,
            "HOSPITAL READMISSION DATA MIGRATION REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Total Files: {migration_results['total_files']}",
            f"Successful Migrations: {migration_results['successful_migrations']}",
            f"Failed Migrations: {migration_results['failed_migrations']}",
            f"Overall Success: {'Yes' if migration_results['overall_success'] else 'No'}",
            "",
            "DETAILED RESULTS:",
            "-" * 40
        ]
        
        for table_name, result in migration_results['migration_details'].items():
            report_lines.extend([
                f"\nTable: {table_name}",
                f"  File: {result['file_path']}",
                f"  Success: {'Yes' if result['migration_success'] else 'No'}"
            ])
            
            if result['validation_results']:
                report_lines.append(f"  Validation: {'Passed' if result['validation_results'].get('is_valid') else 'Failed'}")
                report_lines.append(f"  Total Rows: {result['validation_results'].get('total_rows', 'N/A')}")
            
            if result['transformation_results']:
                report_lines.append(f"  Transformed Rows: {result['transformation_results'].get('transformed_rows', 'N/A')}")
            
            if result['error_message']:
                report_lines.append(f"  Error: {result['error_message']}")
        
        report_lines.extend([
            "",
            "MIGRATION LOG:",
            "-" * 40
        ])
        
        for log_entry in self.migration_log:
            report_lines.append(
                f"{log_entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - "
                f"{log_entry['table_name']}: {'SUCCESS' if log_entry['migration_success'] else 'FAILED'}"
            )
        
        return "\n".join(report_lines)
    
    def save_migration_report(self, migration_results: Dict[str, Any], 
                            output_path: str) -> bool:
        """
        Save migration report to file.
        
        Args:
            migration_results: Migration results
            output_path: Output file path
            
        Returns:
            bool: True if saved successfully
        """
        try:
            report = self.generate_migration_report(migration_results)
            
            with open(output_path, 'w') as f:
                f.write(report)
            
            logger.info(f"Migration report saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save migration report: {str(e)}")
            return False


# Example usage and testing
if __name__ == "__main__":
    # Example migration workflow
    print("Data Migration Module")
    print("Available classes:")
    print("- DataValidator: Data validation")
    print("- DataTransformer: Data transformation")
    print("- DataMigrator: Complete migration workflow")
    
    # Example usage would be:
    # db_manager = DatabaseManager()
    # migrator = DataMigrator(db_manager)
    # 
    # data_files = {
    #     'patients': 'diabetic_data.csv',
    #     'medications': 'medication_data.csv'
    # }
    # 
    # results = migrator.migrate_all_data(data_files)
    # migrator.save_migration_report(results, 'migration_report.txt') 