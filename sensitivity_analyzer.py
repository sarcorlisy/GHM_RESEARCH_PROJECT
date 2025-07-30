"""
Sensitivity Analysis Module for Hospital Readmission Prediction
优化后的敏感性分析模块，支持单个子组分析和批量比较
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Callable
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Import custom modules
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from feature_selector import FeatureSelector
from model_trainer import ModelTrainer
from eda_analyzer import EDAAnalyzer
from pipeline_config import MODEL_CONFIG, DATA_PATHS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sensitivity_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

class SensitivityAnalyzer:
    """
    优化后的敏感性分析器 - 支持单个子组分析和批量比较
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化敏感性分析器
        
        Args:
            config: 配置字典
        """
        self.config = config or MODEL_CONFIG
        self.data_loader = None
        self.preprocessor = None
        self.feature_selector = None
        self.model_trainer = None
        self.eda_analyzer = None
        
        # 存储预处理后的数据（只处理一次）
        self.processed_data = None
        
        # 存储结果
        self.subgroup_results = {}
        
        logger.info("Sensitivity Analyzer initialized")
    
    def load_and_preprocess_data(self) -> pd.DataFrame:
        """
        加载和预处理数据（只运行一次）
        
        Returns:
            预处理后的数据
        """
        if self.processed_data is not None:
            logger.info("Using cached preprocessed data")
            return self.processed_data
        
        logger.info("Loading and preprocessing data...")
        
        # 加载数据
        self.data_loader = DataLoader()
        raw_data = self.data_loader.merge_data()
        
        # 预处理
        self.preprocessor = DataPreprocessor()
        self.processed_data = self.preprocessor.apply_feature_engineering(raw_data)
        self.processed_data = self.preprocessor.prepare_target_variable(self.processed_data)
        
        logger.info(f"Data loaded and preprocessed. Shape: {self.processed_data.shape}")
        return self.processed_data
    
    def define_subgroups(self) -> Dict[str, Dict[str, Any]]:
        """
        定义所有可用的子组
        
        Returns:
            子组定义字典
        """
        subgroups = {
            'comorbidity_1': {
                'name': 'Single Comorbidity (1)',
                'description': 'Patients with exactly 1 comorbidity',
                'filter_condition': lambda df: df['comorbidity'] == 1,
                'expected_size': 'Small subset'
            },
            'comorbidity_2': {
                'name': 'Double Comorbidity (2)',
                'description': 'Patients with exactly 2 comorbidities',
                'filter_condition': lambda df: df['comorbidity'] == 2,
                'expected_size': 'Medium subset'
            },
            'diabetes_diag_all': {
                'name': 'Any Diabetes Diagnosis',
                'description': 'Patients with diabetes in any of the three diagnoses',
                'filter_condition': self._filter_diabetes_any,
                'expected_size': 'Large subset'
            },
            'diabetes_diag1': {
                'name': 'Primary Diabetes Diagnosis',
                'description': 'Patients with diabetes as primary diagnosis',
                'filter_condition': self._filter_diabetes_primary,
                'expected_size': 'Medium subset'
            },
            'elderly_patients': {
                'name': 'Elderly Patients (70+)',
                'description': 'Patients aged 70 years or older',
                'filter_condition': lambda df: df['age'] >= 70,
                'expected_size': 'Medium subset'
            },
            'young_patients': {
                'name': 'Young Patients (<50)',
                'description': 'Patients aged less than 50 years',
                'filter_condition': lambda df: df['age'] < 50,
                'expected_size': 'Small subset'
            },
            'female_patients': {
                'name': 'Female Patients',
                'description': 'Female patients only',
                'filter_condition': lambda df: df['gender'] == 'Female',
                'expected_size': 'Large subset'
            },
            'male_patients': {
                'name': 'Male Patients',
                'description': 'Male patients only',
                'filter_condition': lambda df: df['gender'] == 'Male',
                'expected_size': 'Large subset'
            }
        }
        
        logger.info(f"Defined {len(subgroups)} subgroups for sensitivity analysis")
        return subgroups
    
    def _filter_diabetes_any(self, df: pd.DataFrame) -> pd.Series:
        """筛选任何诊断为糖尿病的患者"""
        def is_diabetes(code):
            try:
                code_str = str(code).strip()
                return code_str.startswith('250')
            except:
                return False
        
        return df.apply(
            lambda row: any([
                is_diabetes(row['diag_1']),
                is_diabetes(row['diag_2']),
                is_diabetes(row['diag_3'])
            ]),
            axis=1
        )
    
    def _filter_diabetes_primary(self, df: pd.DataFrame) -> pd.Series:
        """筛选主诊断为糖尿病的患者"""
        def is_diabetes(code):
            try:
                code_str = str(code).strip()
                return code_str.startswith('250')
            except:
                return False
        
        return df['diag_1'].apply(is_diabetes)
    
    def run_single_subgroup(self, subgroup_name: str, 
                          custom_filter: Optional[Callable] = None,
                          save_results: bool = True) -> Dict[str, Any]:
        """
        运行单个子组分析
        
        Args:
            subgroup_name: 子组名称或自定义名称
            custom_filter: 自定义筛选函数
            save_results: 是否保存结果到文件
            
        Returns:
            子组分析结果
        """
        logger.info(f"Running analysis for subgroup: {subgroup_name}")
        
        # 确保数据已加载
        if self.processed_data is None:
            self.load_and_preprocess_data()
        
        # 获取子组数据
        if custom_filter is not None:
            # 使用自定义筛选
            subgroup_mask = custom_filter(self.processed_data)
            subgroup_data = self.processed_data[subgroup_mask].copy()
        else:
            # 使用预定义的子组
            subgroups = self.define_subgroups()
            if subgroup_name not in subgroups:
                raise ValueError(f"Subgroup '{subgroup_name}' not found. Available: {list(subgroups.keys())}")
            
            subgroup_config = subgroups[subgroup_name]
            subgroup_mask = subgroup_config['filter_condition'](self.processed_data)
            subgroup_data = self.processed_data[subgroup_mask].copy()
        
        if len(subgroup_data) == 0:
            logger.warning(f"Subgroup {subgroup_name} is empty!")
            return {
                'subgroup_name': subgroup_name,
                'data_size': 0,
                'status': 'empty',
                'results': None
            }
        
        logger.info(f"Subgroup {subgroup_name} size: {len(subgroup_data)}")
        
        # 运行完整的pipeline
        try:
            # 数据分割
            X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.split_data(
                subgroup_data,
                test_size=self.config['test_size'],
                val_size=self.config['val_size'],
                random_state=self.config['random_state']
            )
            
            # 编码和标准化
            X_train, X_val, X_test = self.preprocessor.encode_categorical_features(X_train, X_val, X_test)
            X_train, X_val, X_test = self.preprocessor.scale_numerical_features(X_train, X_val, X_test)
            
            # SMOTE平衡
            X_train_balanced, y_train_balanced = self.preprocessor.apply_smote(X_train, y_train)
            
            # 特征选择
            self.feature_selector = FeatureSelector()
            selected_features = self.feature_selector.select_all_features(
                X_train_balanced, y_train_balanced,
                top_n=self.config['feature_selection_top_n']
            )
            
            # 模型训练
            self.model_trainer = ModelTrainer(random_state=self.config['random_state'])
            
            # 使用互信息选择的特征
            best_features = selected_features['MutualInfo']
            X_train_selected = X_train_balanced[best_features]
            X_val_selected = X_val[best_features]
            X_test_selected = X_test[best_features]
            
            # 训练模型
            training_results = self.model_trainer.train_all_models(
                X_train_selected, y_train_balanced, X_val_selected, y_val
            )
            
            # 测试集评估
            test_results = self.model_trainer.evaluate_on_test_set(X_test_selected, y_test)
            
            # 获取最佳模型
            best_model_name, best_model = self.model_trainer.get_best_model('auc')
            
            # 运行EDA分析
            self.eda_analyzer = EDAAnalyzer(subgroup_data)
            
            # 保存结果
            results = {
                'subgroup_name': subgroup_name,
                'data_size': len(subgroup_data),
                'status': 'success',
                'training_results': training_results,
                'test_results': test_results,
                'selected_features': selected_features,
                'best_model': best_model_name,
                'best_model_object': best_model,
                'feature_importance': self.feature_selector.feature_importance_scores if hasattr(self.feature_selector, 'feature_importance_scores') and self.feature_selector.feature_importance_scores else {},
                'eda_results': self._run_eda_analysis(subgroup_data)
            }
            
            # 保存到内存
            self.subgroup_results[subgroup_name] = results
            
            # 保存到文件（如果需要）
            if save_results:
                self._save_subgroup_results(subgroup_name, results)
            
            logger.info(f"Subgroup {subgroup_name} analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in subgroup {subgroup_name} analysis: {e}")
            error_result = {
                'subgroup_name': subgroup_name,
                'data_size': len(subgroup_data),
                'status': 'error',
                'error': str(e),
                'results': None
            }
            self.subgroup_results[subgroup_name] = error_result
            return error_result
    
    def _run_eda_analysis(self, subgroup_data: pd.DataFrame) -> Dict[str, Any]:
        """
        运行EDA分析
        
        Args:
            subgroup_data: 子组数据
            
        Returns:
            EDA分析结果
        """
        try:
            # 基本统计
            basic_stats = {
                'shape': subgroup_data.shape,
                'missing_values': subgroup_data.isnull().sum().to_dict(),
                'target_distribution': subgroup_data['readmitted_binary'].value_counts().to_dict()
            }
            
            # 年龄和性别分析
            age_gender_stats = subgroup_data.groupby(['age_group', 'gender'])['readmitted_binary'].agg(['count', 'mean']).reset_index()
            
            # 合并症分析
            comorbidity_stats = subgroup_data.groupby('comorbidity')['readmitted_binary'].agg(['count', 'mean']).reset_index()
            
            return {
                'basic_stats': basic_stats,
                'age_gender_stats': age_gender_stats,
                'comorbidity_stats': comorbidity_stats
            }
            
        except Exception as e:
            logger.error(f"Error in EDA analysis: {e}")
            return {'error': str(e)}
    
    def compare_subgroups(self, subgroup_names: List[str]) -> Dict[str, Any]:
        """
        比较多个子组的结果
        
        Args:
            subgroup_names: 要比较的子组名称列表
            
        Returns:
            比较结果
        """
        logger.info(f"Comparing subgroups: {subgroup_names}")
        
        # 确保所有子组都已分析
        missing_subgroups = [name for name in subgroup_names if name not in self.subgroup_results]
        if missing_subgroups:
            logger.warning(f"Missing subgroups: {missing_subgroups}. Running analysis...")
            for name in missing_subgroups:
                self.run_single_subgroup(name)
        
        # 生成比较报告
        comparison_report = self._generate_comparison_report(subgroup_names)
        
        # 生成可视化
        self._generate_comparison_visualizations(comparison_report)
        
        return {
            'comparison_report': comparison_report,
            'subgroup_results': {name: self.subgroup_results[name] for name in subgroup_names}
        }
    
    def _generate_comparison_report(self, subgroup_names: List[str]) -> pd.DataFrame:
        """生成子组比较报告"""
        comparison_data = []
        
        for subgroup_name in subgroup_names:
            result = self.subgroup_results[subgroup_name]
            if result['status'] == 'success' and result['test_results'] is not None:
                test_results = result['test_results']
                
                for model_name in test_results.index:
                    row = {
                        'Subgroup': subgroup_name,
                        'Model': model_name,
                        'Data_Size': result['data_size'],
                        'Accuracy': test_results.loc[model_name, 'Accuracy'],
                        'AUC': test_results.loc[model_name, 'AUC'],
                        'F1_Score': test_results.loc[model_name, 'F1_Score'],
                        'Precision': test_results.loc[model_name, 'Precision'],
                        'Recall': test_results.loc[model_name, 'Recall']
                    }
                    comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df
    
    def _generate_comparison_visualizations(self, comparison_report: pd.DataFrame) -> None:
        """生成比较可视化"""
        if comparison_report.empty:
            return
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        
        # 1. AUC比较图
        plt.figure(figsize=(12, 6))
        pivot_auc = comparison_report.pivot(index='Subgroup', columns='Model', values='AUC')
        pivot_auc.plot(kind='bar', ax=plt.gca())
        plt.title('AUC Comparison Across Subgroups and Models')
        plt.ylabel('AUC Score')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        
        # 2. F1 Score比较图
        plt.figure(figsize=(12, 6))
        pivot_f1 = comparison_report.pivot(index='Subgroup', columns='Model', values='F1_Score')
        pivot_f1.plot(kind='bar', ax=plt.gca())
        plt.title('F1 Score Comparison Across Subgroups and Models')
        plt.ylabel('F1 Score')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        
        # 3. 数据大小比较
        plt.figure(figsize=(10, 6))
        data_sizes = comparison_report.groupby('Subgroup')['Data_Size'].first()
        data_sizes.plot(kind='bar')
        plt.title('Data Size Comparison Across Subgroups')
        plt.ylabel('Number of Patients')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def _save_subgroup_results(self, subgroup_name: str, result: Dict[str, Any]) -> None:
        """保存单个子组的结果"""
        output_dir = Path(DATA_PATHS['output_dir']) / 'sensitivity_analysis' / subgroup_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存测试结果
        if result['status'] == 'success' and result['test_results'] is not None:
            result['test_results'].to_csv(output_dir / 'test_results.csv')
        
        # 保存特征选择结果
        if result['status'] == 'success' and result['selected_features'] is not None:
            with open(output_dir / 'selected_features.json', 'w') as f:
                json.dump(result['selected_features'], f, indent=2)
        
        # 保存EDA结果
        if result['status'] == 'success' and result['eda_results'] is not None:
            with open(output_dir / 'eda_results.json', 'w') as f:
                json.dump(result['eda_results'], f, indent=2, default=str)
    
    def get_available_subgroups(self) -> List[str]:
        """获取所有可用的子组名称"""
        return list(self.define_subgroups().keys())
    
    def get_subgroup_info(self, subgroup_name: str) -> Dict[str, Any]:
        """获取子组信息"""
        subgroups = self.define_subgroups()
        if subgroup_name in subgroups:
            return subgroups[subgroup_name]
        else:
            raise ValueError(f"Subgroup '{subgroup_name}' not found")

def main():
    """主函数"""
    analyzer = SensitivityAnalyzer()
    print("Sensitivity Analyzer initialized successfully!")
    print(f"Available subgroups: {analyzer.get_available_subgroups()}")

if __name__ == "__main__":
    main() 