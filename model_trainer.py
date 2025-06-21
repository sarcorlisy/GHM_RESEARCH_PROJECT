"""
Model Training Module for Hospital Readmission Prediction Pipeline
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report)
import logging
from typing import Dict, List, Tuple, Any
import joblib
from pathlib import Path
import warnings

from pipeline_config import MODEL_CONFIG, MODELS

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

class ModelTrainer:
    """模型训练器类，负责训练和评估多种机器学习模型"""
    
    def __init__(self, random_state: int = 42):
        """
        初始化模型训练器
        
        Args:
            random_state: 随机种子
        """
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.cv_results = {}
        self.test_results = {}
        
    def get_models(self) -> Dict[str, Any]:
        """
        获取所有可用的模型
        
        Returns:
            模型字典
        """
        return {
            'LogisticRegression': LogisticRegression(solver='liblinear', random_state=self.random_state),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'XGBoost': XGBClassifier(eval_metric='logloss', random_state=self.random_state)
        }
    
    def evaluate_model_with_cv(self, model: Any, X: pd.DataFrame, y: pd.Series, 
                              cv_folds: int = 5) -> Tuple[float, float]:
        """
        使用交叉验证评估模型
        
        Args:
            model: 机器学习模型
            X: 特征矩阵
            y: 目标变量
            cv_folds: 交叉验证折数
            
        Returns:
            AUC和F1分数的平均值
        """
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        auc_scores = []
        f1_scores = []
        
        for train_idx, test_idx in skf.split(X, y):
            X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
            y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train_cv, y_train_cv)
            y_pred = model.predict(X_test_cv)
            y_prob = model.predict_proba(X_test_cv)[:, 1]
            
            auc_scores.append(roc_auc_score(y_test_cv, y_prob))
            f1_scores.append(f1_score(y_test_cv, y_pred))
        
        return np.mean(auc_scores), np.mean(f1_scores)
    
    def train_single_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict[str, Any]:
        """
        训练单个模型
        
        Args:
            model_name: 模型名称
            X_train: 训练集特征
            y_train: 训练集目标变量
            X_val: 验证集特征
            y_val: 验证集目标变量
            
        Returns:
            训练结果字典
        """
        logger.info(f"Training {model_name}...")
        
        models = self.get_models()
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(models.keys())}")
        
        model = models[model_name]
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 交叉验证评估
        auc_cv, f1_cv = self.evaluate_model_with_cv(model, X_train, y_train, 
                                                   cv_folds=MODEL_CONFIG['cv_folds'])
        
        # 验证集评估（如果提供）
        val_results = {}
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            y_val_prob = model.predict_proba(X_val)[:, 1]
            
            val_results = {
                'accuracy': accuracy_score(y_val, y_val_pred),
                'precision': precision_score(y_val, y_val_pred),
                'recall': recall_score(y_val, y_val_pred),
                'f1': f1_score(y_val, y_val_pred),
                'auc': roc_auc_score(y_val, y_val_prob)
            }
        
        # 保存训练好的模型
        self.trained_models[model_name] = model
        
        results = {
            'model_name': model_name,
            'cv_auc': auc_cv,
            'cv_f1': f1_cv,
            'val_results': val_results
        }
        
        logger.info(f"{model_name} training completed - CV AUC: {auc_cv:.3f}, CV F1: {f1_cv:.3f}")
        return results
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame = None, y_val: pd.Series = None) -> pd.DataFrame:
        """
        训练所有模型
        
        Args:
            X_train: 训练集特征
            y_train: 训练集目标变量
            X_val: 验证集特征
            y_val: 验证集目标变量
            
        Returns:
            所有模型的训练结果DataFrame
        """
        logger.info("Training all models...")
        
        models = self.get_models()
        results = []
        
        for model_name in models.keys():
            try:
                result = self.train_single_model(model_name, X_train, y_train, X_val, y_val)
                results.append(result)
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                results.append({
                    'model_name': model_name,
                    'cv_auc': 0.0,
                    'cv_f1': 0.0,
                    'val_results': {}
                })
        
        results_df = pd.DataFrame(results)
        logger.info("All models training completed")
        
        return results_df
    
    def evaluate_on_test_set(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        在测试集上评估所有训练好的模型
        
        Args:
            X_test: 测试集特征
            y_test: 测试集目标变量
            
        Returns:
            测试集评估结果DataFrame
        """
        logger.info("Evaluating models on test set...")
        
        if not self.trained_models:
            logger.warning("No trained models available. Train models first.")
            return pd.DataFrame()
        
        test_results = []
        
        for model_name, model in self.trained_models.items():
            try:
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                
                results = {
                    'model_name': model_name,
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred),
                    'auc': roc_auc_score(y_test, y_prob)
                }
                
                test_results.append(results)
                self.test_results[model_name] = results
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name} on test set: {e}")
        
        test_results_df = pd.DataFrame(test_results)
        logger.info("Test set evaluation completed")
        
        return test_results_df
    
    def get_best_model(self, metric: str = 'auc') -> Tuple[str, Any]:
        """
        获取最佳模型
        
        Args:
            metric: 评估指标
            
        Returns:
            最佳模型名称和模型对象
        """
        if not self.test_results:
            logger.warning("No test results available. Run evaluate_on_test_set first.")
            return None, None
        
        best_model_name = max(self.test_results.keys(), 
                            key=lambda x: self.test_results[x][metric])
        best_model = self.trained_models[best_model_name]
        
        logger.info(f"Best model by {metric}: {best_model_name} ({self.test_results[best_model_name][metric]:.3f})")
        return best_model_name, best_model
    
    def save_models(self, output_dir: str = 'outputs/models') -> None:
        """
        保存训练好的模型
        
        Args:
            output_dir: 输出目录
        """
        if not self.trained_models:
            logger.warning("No trained models to save.")
            return
        
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.trained_models.items():
            model_path = f"{output_dir}/{model_name}.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Model {model_name} saved to {model_path}")
    
    def load_models(self, models_dir: str = 'outputs/models') -> None:
        """
        加载保存的模型
        
        Args:
            models_dir: 模型目录
        """
        models_path = Path(models_dir)
        if not models_path.exists():
            logger.warning(f"Models directory {models_dir} does not exist.")
            return
        
        for model_file in models_path.glob("*.joblib"):
            model_name = model_file.stem
            model = joblib.load(model_file)
            self.trained_models[model_name] = model
            logger.info(f"Model {model_name} loaded from {model_file}")
    
    def generate_model_report(self, output_path: str = 'outputs/model_report.txt') -> None:
        """
        生成模型训练报告
        
        Args:
            output_path: 报告输出路径
        """
        if not self.test_results:
            logger.warning("No test results available for report generation.")
            return
        
        # 创建输出目录
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("Hospital Readmission Prediction - Model Training Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Model Performance Summary:\n")
            f.write("-" * 30 + "\n")
            
            for model_name, results in self.test_results.items():
                f.write(f"\n{model_name}:\n")
                for metric, value in results.items():
                    if metric != 'model_name':
                        f.write(f"  {metric}: {value:.3f}\n")
            
            # 找出最佳模型
            best_auc_model = max(self.test_results.keys(), 
                               key=lambda x: self.test_results[x]['auc'])
            best_f1_model = max(self.test_results.keys(), 
                              key=lambda x: self.test_results[x]['f1'])
            
            f.write(f"\nBest Model by AUC: {best_auc_model} ({self.test_results[best_auc_model]['auc']:.3f})\n")
            f.write(f"Best Model by F1: {best_f1_model} ({self.test_results[best_f1_model]['f1']:.3f})\n")
        
        logger.info(f"Model report generated: {output_path}")
    
    def plot_model_comparison(self, save_path: str = 'outputs/model_comparison.png') -> None:
        """
        绘制模型比较图
        
        Args:
            save_path: 保存路径
        """
        if not self.test_results:
            logger.warning("No test results available for plotting.")
            return
        
        import matplotlib.pyplot as plt
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        models = list(self.test_results.keys())
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4))
        
        for i, metric in enumerate(metrics):
            values = [self.test_results[model][metric] for model in models]
            
            axes[i].bar(models, values)
            axes[i].set_title(f'{metric.upper()}')
            axes[i].set_ylabel('Score')
            axes[i].tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison plot saved to: {save_path}")
        plt.show()

def main():
    """主函数，用于测试模型训练功能"""
    from data_loader import DataLoader
    from data_preprocessor import DataPreprocessor
    from feature_selector import FeatureSelector
    
    # 加载和预处理数据
    loader = DataLoader()
    df = loader.merge_data()
    
    preprocessor = DataPreprocessor()
    df = preprocessor.apply_feature_engineering(df)
    df = preprocessor.prepare_target_variable(df)
    
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(df)
    X_train, X_val, X_test = preprocessor.encode_categorical_features(X_train, X_val, X_test)
    X_train, X_val, X_test = preprocessor.scale_numerical_features(X_train, X_val, X_test)
    X_train_balanced, y_train_balanced = preprocessor.apply_smote(X_train, y_train)
    
    # 特征选择
    feature_selector = FeatureSelector()
    selected_features = feature_selector.select_all_features(X_train_balanced, y_train_balanced, top_n=15)
    
    # 使用最佳特征集（这里使用互信息方法选择的特征）
    best_features = selected_features['MutualInfo']
    X_train_selected = X_train_balanced[best_features]
    X_val_selected = X_val[best_features]
    X_test_selected = X_test[best_features]
    
    # 初始化模型训练器
    trainer = ModelTrainer()
    
    # 训练所有模型
    training_results = trainer.train_all_models(X_train_selected, y_train_balanced, 
                                               X_val_selected, y_val)
    
    print("Training Results:")
    print(training_results)
    
    # 在测试集上评估
    test_results = trainer.evaluate_on_test_set(X_test_selected, y_test)
    
    print("\nTest Results:")
    print(test_results)
    
    # 获取最佳模型
    best_model_name, best_model = trainer.get_best_model('auc')
    
    # 保存模型和生成报告
    trainer.save_models()
    trainer.generate_model_report()
    trainer.plot_model_comparison()
    
    return trainer, test_results

if __name__ == "__main__":
    main() 