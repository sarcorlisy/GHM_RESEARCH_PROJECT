"""
Main Pipeline for Hospital Readmission Prediction
整合所有模块的端到端数据科学pipeline
"""
import pandas as pd
import numpy as np
import logging
import argparse
from pathlib import Path
import warnings
from typing import Dict, Any, Optional

# 导入自定义模块
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from feature_selector import FeatureSelector
from model_trainer import ModelTrainer
from pipeline_config import MODEL_CONFIG, DATA_PATHS

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

class HospitalReadmissionPipeline:
    """医院再入院预测主pipeline类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化pipeline
        
        Args:
            config: 配置字典
        """
        self.config = config or MODEL_CONFIG
        self.data_loader = None
        self.preprocessor = None
        self.feature_selector = None
        self.model_trainer = None
        
        # 存储中间结果
        self.raw_data = None
        self.processed_data = None
        self.selected_features = None
        self.training_results = None
        self.test_results = None
        
        logger.info("Hospital Readmission Pipeline initialized")
    
    def run_data_loading(self) -> pd.DataFrame:
        """
        运行数据加载步骤
        
        Returns:
            合并后的原始数据
        """
        logger.info("=" * 50)
        logger.info("STEP 1: Data Loading")
        logger.info("=" * 50)
        
        self.data_loader = DataLoader()
        self.raw_data = self.data_loader.merge_data()
        
        # 获取数据信息
        data_info = self.data_loader.get_data_info()
        logger.info(f"Data loaded successfully: {data_info['shape']}")
        logger.info(f"Number of features: {len(data_info['columns'])}")
        
        # 保存合并后的数据
        self.data_loader.save_merged_data()
        
        return self.raw_data
    
    def run_data_preprocessing(self) -> tuple:
        """
        运行数据预处理步骤
        
        Returns:
            预处理后的训练、验证、测试数据
        """
        logger.info("=" * 50)
        logger.info("STEP 2: Data Preprocessing")
        logger.info("=" * 50)
        
        self.preprocessor = DataPreprocessor()
        
        # 应用特征工程
        self.processed_data = self.preprocessor.apply_feature_engineering(self.raw_data)
        logger.info(f"Feature engineering completed. New shape: {self.processed_data.shape}")
        
        # 准备目标变量
        self.processed_data = self.preprocessor.prepare_target_variable(self.processed_data)
        
        # 分割数据
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.split_data(
            self.processed_data,
            test_size=self.config['test_size'],
            val_size=self.config['val_size'],
            random_state=self.config['random_state']
        )
        
        # 编码分类特征
        X_train, X_val, X_test = self.preprocessor.encode_categorical_features(X_train, X_val, X_test)
        
        # 标准化数值特征
        X_train, X_val, X_test = self.preprocessor.scale_numerical_features(X_train, X_val, X_test)
        
        # 应用SMOTE平衡数据集
        X_train_balanced, y_train_balanced = self.preprocessor.apply_smote(X_train, y_train)
        
        # 保存预处理后的数据
        self.preprocessor.save_preprocessed_data(
            X_train_balanced, X_val, X_test, 
            y_train_balanced, y_val, y_test
        )
        
        logger.info("Data preprocessing completed successfully")
        return X_train_balanced, X_val, X_test, y_train_balanced, y_val, y_test
    
    def run_feature_selection(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, list]:
        """
        运行特征选择步骤
        
        Args:
            X_train: 训练集特征
            y_train: 训练集目标变量
            
        Returns:
            选择的特征字典
        """
        logger.info("=" * 50)
        logger.info("STEP 3: Feature Selection")
        logger.info("=" * 50)
        
        self.feature_selector = FeatureSelector()
        
        # 使用所有方法选择特征
        self.selected_features = self.feature_selector.select_all_features(
            X_train, y_train, 
            top_n=self.config['feature_selection_top_n']
        )
        
        # 打印选择结果
        for method, features in self.selected_features.items():
            logger.info(f"{method} selected {len(features)} features")
        
        # 获取共同特征
        common_features = self.feature_selector.get_common_features(min_methods=2)
        logger.info(f"Common features selected by at least 2 methods: {len(common_features)}")
        
        # 保存选择的特征
        self.feature_selector.save_selected_features()
        
        # 绘制特征重要性
        self.feature_selector.plot_feature_importance(save_path="outputs/feature_importance.png")
        
        return self.selected_features
    
    def run_model_training(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series,
                          selected_features: Dict[str, list]) -> tuple:
        """
        运行模型训练步骤
        
        Args:
            X_train: 训练集特征
            y_train: 训练集目标变量
            X_val: 验证集特征
            y_val: 验证集目标变量
            X_test: 测试集特征
            y_test: 测试集目标变量
            selected_features: 选择的特征字典
            
        Returns:
            训练结果和测试结果
        """
        logger.info("=" * 50)
        logger.info("STEP 4: Model Training")
        logger.info("=" * 50)
        
        self.model_trainer = ModelTrainer(random_state=self.config['random_state'])
        
        # 使用最佳特征集（这里使用互信息方法选择的特征）
        best_features = selected_features['MutualInfo']
        logger.info(f"Using {len(best_features)} features selected by Mutual Information")
        
        X_train_selected = X_train[best_features]
        X_val_selected = X_val[best_features]
        X_test_selected = X_test[best_features]
        
        # 训练所有模型
        self.training_results = self.model_trainer.train_all_models(
            X_train_selected, y_train, X_val_selected, y_val
        )
        
        logger.info("Training Results:")
        logger.info(self.training_results.to_string())
        
        # 在测试集上评估
        self.test_results = self.model_trainer.evaluate_on_test_set(X_test_selected, y_test)
        
        logger.info("Test Results:")
        logger.info(self.test_results.to_string())
        
        # 获取最佳模型
        best_model_name, best_model = self.model_trainer.get_best_model('auc')
        
        # 保存模型和生成报告
        self.model_trainer.save_models()
        self.model_trainer.generate_model_report()
        self.model_trainer.plot_model_comparison()
        
        return self.training_results, self.test_results
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        运行完整的pipeline
        
        Returns:
            包含所有结果的字典
        """
        logger.info("Starting Hospital Readmission Prediction Pipeline")
        logger.info("=" * 60)
        
        try:
            # 步骤1: 数据加载
            self.run_data_loading()
            
            # 步骤2: 数据预处理
            X_train, X_val, X_test, y_train, y_val, y_test = self.run_data_preprocessing()
            
            # 步骤3: 特征选择
            selected_features = self.run_feature_selection(X_train, y_train)
            
            # 步骤4: 模型训练
            training_results, test_results = self.run_model_training(
                X_train, X_val, X_test, y_train, y_val, y_test, selected_features
            )
            
            # 生成最终报告
            self.generate_final_report()
            
            logger.info("Pipeline completed successfully!")
            
            return {
                'raw_data': self.raw_data,
                'processed_data': self.processed_data,
                'selected_features': selected_features,
                'training_results': training_results,
                'test_results': test_results,
                'best_model': self.model_trainer.get_best_model('auc')
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            raise
    
    def generate_final_report(self) -> None:
        """生成最终的综合报告"""
        logger.info("=" * 50)
        logger.info("Generating Final Report")
        logger.info("=" * 50)
        
        report_path = "outputs/final_pipeline_report.txt"
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Hospital Readmission Prediction - Final Pipeline Report\n")
            f.write("=" * 60 + "\n\n")
            
            # 数据信息
            f.write("1. DATA INFORMATION\n")
            f.write("-" * 20 + "\n")
            f.write(f"Raw data shape: {self.raw_data.shape}\n")
            f.write(f"Processed data shape: {self.processed_data.shape}\n")
            f.write(f"Number of features after engineering: {len(self.processed_data.columns)}\n\n")
            
            # 特征选择信息
            f.write("2. FEATURE SELECTION RESULTS\n")
            f.write("-" * 30 + "\n")
            for method, features in self.selected_features.items():
                f.write(f"{method}: {len(features)} features selected\n")
            f.write("\n")
            
            # 模型性能
            f.write("3. MODEL PERFORMANCE\n")
            f.write("-" * 20 + "\n")
            f.write("Test Set Results:\n")
            f.write(self.test_results.to_string())
            f.write("\n\n")
            
            # 最佳模型
            best_model_name, best_model = self.model_trainer.get_best_model('auc')
            f.write(f"Best Model (by AUC): {best_model_name}\n")
            f.write(f"Best AUC Score: {self.test_results.loc[self.test_results['model_name'] == best_model_name, 'auc'].iloc[0]:.3f}\n")
            
            # 特征重要性
            f.write("\n4. TOP FEATURES BY IMPORTANCE\n")
            f.write("-" * 30 + "\n")
            for method, features in self.selected_features.items():
                f.write(f"\n{method} Top 5 Features:\n")
                for i, feature in enumerate(features[:5], 1):
                    f.write(f"  {i}. {feature}\n")
        
        logger.info(f"Final report generated: {report_path}")
    
    def load_and_predict(self, new_data_path: str, model_name: str = None) -> pd.DataFrame:
        """
        加载模型并对新数据进行预测
        
        Args:
            new_data_path: 新数据文件路径
            model_name: 要使用的模型名称，如果为None则使用最佳模型
            
        Returns:
            预测结果DataFrame
        """
        logger.info("Loading model and making predictions on new data...")
        
        # 加载模型
        if self.model_trainer is None:
            self.model_trainer = ModelTrainer()
            self.model_trainer.load_models()
        
        # 获取要使用的模型
        if model_name is None:
            model_name, model = self.model_trainer.get_best_model('auc')
        else:
            model = self.model_trainer.trained_models.get(model_name)
            if model is None:
                raise ValueError(f"Model {model_name} not found")
        
        # 加载新数据
        new_data = pd.read_csv(new_data_path)
        
        # 应用相同的预处理
        preprocessor = DataPreprocessor()
        new_data = preprocessor.apply_feature_engineering(new_data)
        
        # 编码和标准化（这里需要加载之前保存的编码器和标准化器）
        # 注意：在实际应用中，需要保存和加载这些转换器
        
        # 进行预测
        predictions = model.predict(new_data)
        probabilities = model.predict_proba(new_data)[:, 1]
        
        # 创建结果DataFrame
        results = pd.DataFrame({
            'patient_id': new_data.get('patient_nbr', range(len(new_data))),
            'readmission_prediction': predictions,
            'readmission_probability': probabilities
        })
        
        logger.info(f"Predictions completed for {len(new_data)} patients")
        return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Hospital Readmission Prediction Pipeline')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--predict', type=str, help='Path to new data for prediction')
    parser.add_argument('--model', type=str, help='Model name for prediction')
    
    args = parser.parse_args()
    
    # 初始化pipeline
    pipeline = HospitalReadmissionPipeline()
    
    if args.predict:
        # 预测模式
        results = pipeline.load_and_predict(args.predict, args.model)
        print("Prediction Results:")
        print(results)
        results.to_csv('outputs/predictions.csv', index=False)
    else:
        # 训练模式
        results = pipeline.run_full_pipeline()
        print("Pipeline completed successfully!")
        print(f"Best model: {results['best_model'][0]}")
        print(f"Best AUC: {results['test_results'].loc[results['test_results']['model_name'] == results['best_model'][0], 'auc'].iloc[0]:.3f}")

if __name__ == "__main__":
    main() 