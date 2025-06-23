"""
Data Loading Module for Hospital Readmission Prediction Pipeline
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional

from pipeline_config import DATA_PATHS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Data loader class, responsible for reading and preprocessing raw data"""
    
    def __init__(self, data_paths: Dict[str, Path] = None):
        """
        Initializes the data loader
        
        Args:
            data_paths: Dictionary of data file paths
        """
        self.data_paths = data_paths or DATA_PATHS
        self.diabetic_data = None
        self.ids_mapping = None
        self.merged_data = None
        
    def load_diabetic_data(self) -> pd.DataFrame:
        """
        Loads the diabetic dataset
        
        Returns:
            Diabetic data as a DataFrame
        """
        logger.info("Loading diabetic data...")
        try:
            self.diabetic_data = pd.read_csv(self.data_paths['diabetic_data'])
            logger.info(f"Diabetic data loaded: {self.diabetic_data.shape}")
            return self.diabetic_data
        except FileNotFoundError:
            logger.error(f"Diabetic data file not found: {self.data_paths['diabetic_data']}")
            raise
    
    def load_ids_mapping(self) -> pd.DataFrame:
        """
        Loads the ID mapping data
        
        Returns:
            ID mapping data as a DataFrame
        """
        logger.info("Loading ID mapping data...")
        try:
            self.ids_mapping = pd.read_csv(self.data_paths['ids_mapping'])
            logger.info(f"ID mapping data loaded: {self.ids_mapping.shape}")
            return self.ids_mapping
        except FileNotFoundError:
            logger.error(f"ID mapping file not found: {self.data_paths['ids_mapping']}")
            raise
    
    def split_ids_mapping(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits the ID mapping data into three separate mapping tables
        
        Returns:
            Tuple containing admission type, discharge disposition, and admission source mapping tables
        """
        if self.ids_mapping is None:
            self.load_ids_mapping()
        
        # Split data based on known row indices
        admission_type_df = self.ids_mapping.iloc[0:8].copy()
        discharge_disposition_df = self.ids_mapping.iloc[10:40].reset_index(drop=True).copy()
        admission_source_df = self.ids_mapping.iloc[42:].reset_index(drop=True).copy()
        
        # Set column names
        admission_type_df.columns = ['admission_type_id', 'admission_type_desc']
        discharge_disposition_df.columns = ['discharge_disposition_id', 'discharge_disposition_desc']
        admission_source_df.columns = ['admission_source_id', 'admission_source_desc']
        
        # Convert ID columns to integer type
        admission_type_df['admission_type_id'] = admission_type_df['admission_type_id'].astype(int)
        discharge_disposition_df['discharge_disposition_id'] = discharge_disposition_df['discharge_disposition_id'].astype(int)
        admission_source_df['admission_source_id'] = admission_source_df['admission_source_id'].astype(int)
        
        logger.info("ID mapping data split into three tables")
        return admission_type_df, discharge_disposition_df, admission_source_df
    
    def merge_data(self) -> pd.DataFrame:
        """
        Merges all data tables
        
        Returns:
            The complete merged dataset
        """
        logger.info("Merging all data tables...")
        
        if self.diabetic_data is None:
            self.load_diabetic_data()
        
        # Get mapping tables
        admission_type_df, discharge_disposition_df, admission_source_df = self.split_ids_mapping()
        
        # Merge data
        merged_df = self.diabetic_data.merge(admission_type_df, on='admission_type_id', how='left')
        merged_df = merged_df.merge(discharge_disposition_df, on='discharge_disposition_id', how='left')
        merged_df = merged_df.merge(admission_source_df, on='admission_source_id', how='left')
        
        # Drop old ID columns to avoid redundancy, but keep 'discharge_disposition_id' for later processing
        cols_to_drop = ['admission_type_id', 'admission_source_id']
        merged_df = merged_df.drop(columns=cols_to_drop)
        
        self.merged_data = merged_df
        logger.info(f"Data merged successfully: {merged_df.shape}")
        return merged_df
    
    def get_data_info(self) -> Dict:
        """
        Gets basic information about the dataset
        
        Returns:
            A dictionary of dataset information
        """
        if self.merged_data is None:
            self.merge_data()
        
        info = {
            'shape': self.merged_data.shape,
            'columns': list(self.merged_data.columns),
            'dtypes': self.merged_data.dtypes.to_dict(),
            'missing_values': self.merged_data.isnull().sum().to_dict(),
            'missing_percentage': (self.merged_data.isnull().sum() / len(self.merged_data) * 100).to_dict()
        }
        
        return info
    
    def save_merged_data(self, output_path: Optional[Path] = None) -> Path:
        """
        Saves the merged data
        
        Args:
            output_path: Output path. If None, a default path is used.
            
        Returns:
            The path to the saved file
        """
        if self.merged_data is None:
            self.merge_data()
        
        if output_path is None:
            output_path = self.data_paths['output_dir'] / 'merged_data.csv'
        
        self.merged_data.to_csv(output_path, index=False)
        logger.info(f"Merged data saved to: {output_path}")
        return output_path

def main():
    """Main function to test data loading functionality"""
    loader = DataLoader()
    
    # Load and merge data
    merged_data = loader.merge_data()
    
    # Get data info
    info = loader.get_data_info()
    print(f"Dataset shape: {info['shape']}")
    print(f"Number of columns: {len(info['columns'])}")
    print(f"Missing values summary:")
    for col, missing in info['missing_percentage'].items():
        if missing > 0:
            print(f"  {col}: {missing:.2f}%")
    
    # Save merged data
    loader.save_merged_data()
    
    return merged_data

if __name__ == "__main__":
    main() 