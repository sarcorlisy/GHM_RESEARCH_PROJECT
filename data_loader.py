"""
Data Loading Module for Hospital Readmission Prediction Pipeline
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional

from pipeline_config import DATA_PATHS

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """数据加载器类，负责读取和预处理原始数据"""
    
    def __init__(self, data_paths: Dict[str, Path] = None):
        """
        初始化数据加载器
        
        Args:
            data_paths: 数据文件路径字典
        """
        self.data_paths = data_paths or DATA_PATHS
        self.diabetic_data = None
        self.ids_mapping = None
        self.merged_data = None
        
    def load_diabetic_data(self) -> pd.DataFrame:
        """
        加载糖尿病数据集
        
        Returns:
            糖尿病数据DataFrame
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
        加载ID映射数据
        
        Returns:
            ID映射数据DataFrame
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
        将ID映射数据分割为三个独立的映射表
        
        Returns:
            入院类型、出院处置、入院来源映射表
        """
        if self.ids_mapping is None:
            self.load_ids_mapping()
        
        # 根据已知的行索引分割数据
        admission_type_df = self.ids_mapping.iloc[0:8].copy()
        discharge_disposition_df = self.ids_mapping.iloc[10:40].reset_index(drop=True).copy()
        admission_source_df = self.ids_mapping.iloc[42:].reset_index(drop=True).copy()
        
        # 设置列名
        admission_type_df.columns = ['admission_type_id', 'admission_type_desc']
        discharge_disposition_df.columns = ['discharge_disposition_id', 'discharge_disposition_desc']
        admission_source_df.columns = ['admission_source_id', 'admission_source_desc']
        
        # 转换ID列为整数类型
        admission_type_df['admission_type_id'] = admission_type_df['admission_type_id'].astype(int)
        discharge_disposition_df['discharge_disposition_id'] = discharge_disposition_df['discharge_disposition_id'].astype(int)
        admission_source_df['admission_source_id'] = admission_source_df['admission_source_id'].astype(int)
        
        logger.info("ID mapping data split into three tables")
        return admission_type_df, discharge_disposition_df, admission_source_df
    
    def merge_data(self) -> pd.DataFrame:
        """
        合并所有数据表
        
        Returns:
            合并后的完整数据集
        """
        logger.info("Merging all data tables...")
        
        if self.diabetic_data is None:
            self.load_diabetic_data()
        
        # 获取映射表
        admission_type_df, discharge_disposition_df, admission_source_df = self.split_ids_mapping()
        
        # 合并数据
        merged_df = self.diabetic_data.merge(admission_type_df, on='admission_type_id', how='left')
        merged_df = merged_df.merge(discharge_disposition_df, on='discharge_disposition_id', how='left')
        merged_df = merged_df.merge(admission_source_df, on='admission_source_id', how='left')
        
        # 删除旧的ID列以避免冗余，但保留'discharge_disposition_id'用于后续处理
        cols_to_drop = ['admission_type_id', 'admission_source_id']
        merged_df = merged_df.drop(columns=cols_to_drop)
        
        self.merged_data = merged_df
        logger.info(f"Data merged successfully: {merged_df.shape}")
        return merged_df
    
    def get_data_info(self) -> Dict:
        """
        获取数据集基本信息
        
        Returns:
            数据集信息字典
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
        保存合并后的数据
        
        Args:
            output_path: 输出路径，如果为None则使用默认路径
            
        Returns:
            保存的文件路径
        """
        if self.merged_data is None:
            self.merge_data()
        
        if output_path is None:
            output_path = self.data_paths['output_dir'] / 'merged_data.csv'
        
        self.merged_data.to_csv(output_path, index=False)
        logger.info(f"Merged data saved to: {output_path}")
        return output_path

def main():
    """主函数，用于测试数据加载功能"""
    loader = DataLoader()
    
    # 加载和合并数据
    merged_data = loader.merge_data()
    
    # 获取数据信息
    info = loader.get_data_info()
    print(f"Dataset shape: {info['shape']}")
    print(f"Number of columns: {len(info['columns'])}")
    print(f"Missing values summary:")
    for col, missing in info['missing_percentage'].items():
        if missing > 0:
            print(f"  {col}: {missing:.2f}%")
    
    # 保存合并后的数据
    loader.save_merged_data()
    
    return merged_data

if __name__ == "__main__":
    main() 