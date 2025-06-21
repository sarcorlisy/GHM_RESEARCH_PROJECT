"""
Feature Selection Module for Hospital Readmission Prediction Pipeline
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
import logging
from typing import Dict, List, Tuple, Callable
import json
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureSelector:
    """特征选择器类，提供多种特征选择方法"""
    
    def __init__(self):
        """初始化特征选择器"""
        self.selected_features = {}
        self.feature_importance_scores = {}
        
    def select_features_by_l1(self, X: pd.DataFrame, y: pd.Series, top_n: int = 15) -> List[str]:
        """
        使用L1正则化进行特征选择
        
        Args:
            X: 特征矩阵
            y: 目标变量
            top_n: 选择的特征数量
            
        Returns:
            选择的特征列表
        """
        logger.info(f"Selecting top {top_n} features using L1 regularization...")
        
        clf = LogisticRegression(penalty='l1', solver='liblinear', random_state=42, max_iter=1000)
        clf.fit(X, y)
        
        coefs = np.abs(clf.coef_[0])
        feature_ranking = pd.Series(coefs, index=X.columns).sort_values(ascending=False)
        
        selected_features = feature_ranking.head(top_n).index.tolist()
        self.feature_importance_scores['L1'] = feature_ranking.to_dict()
        
        logger.info(f"L1 feature selection completed. Selected features: {selected_features}")
        return selected_features
    
    def select_features_by_mi(self, X: pd.DataFrame, y: pd.Series, top_n: int = 15) -> List[str]:
        """
        使用互信息进行特征选择
        
        Args:
            X: 特征矩阵
            y: 目标变量
            top_n: 选择的特征数量
            
        Returns:
            选择的特征列表
        """
        logger.info(f"Selecting top {top_n} features using Mutual Information...")
        
        mi_scores = mutual_info_classif(X, y, random_state=42)
        feature_ranking = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        
        selected_features = feature_ranking.head(top_n).index.tolist()
        self.feature_importance_scores['MutualInfo'] = feature_ranking.to_dict()
        
        logger.info(f"Mutual Information feature selection completed. Selected features: {selected_features}")
        return selected_features
    
    def select_features_by_tree(self, X: pd.DataFrame, y: pd.Series, top_n: int = 15) -> List[str]:
        """
        使用树模型特征重要性进行特征选择
        
        Args:
            X: 特征矩阵
            y: 目标变量
            top_n: 选择的特征数量
            
        Returns:
            选择的特征列表
        """
        logger.info(f"Selecting top {top_n} features using Tree-based importance...")
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        
        importances = clf.feature_importances_
        feature_ranking = pd.Series(importances, index=X.columns).sort_values(ascending=False)
        
        selected_features = feature_ranking.head(top_n).index.tolist()
        self.feature_importance_scores['TreeImportance'] = feature_ranking.to_dict()
        
        logger.info(f"Tree-based feature selection completed. Selected features: {selected_features}")
        return selected_features
    
    def get_feature_selectors(self) -> Dict[str, Callable]:
        """
        获取所有特征选择方法
        
        Returns:
            特征选择方法字典
        """
        return {
            'L1': self.select_features_by_l1,
            'MutualInfo': self.select_features_by_mi,
            'TreeImportance': self.select_features_by_tree
        }
    
    def select_features_by_method(self, method: str, X: pd.DataFrame, y: pd.Series, 
                                 top_n: int = 15) -> List[str]:
        """
        根据指定方法选择特征
        
        Args:
            method: 特征选择方法名称
            X: 特征矩阵
            y: 目标变量
            top_n: 选择的特征数量
            
        Returns:
            选择的特征列表
        """
        selectors = self.get_feature_selectors()
        
        if method not in selectors:
            raise ValueError(f"Unknown feature selection method: {method}. Available methods: {list(selectors.keys())}")
        
        return selectors[method](X, y, top_n)
    
    def select_all_features(self, X: pd.DataFrame, y: pd.Series, top_n: int = 15) -> Dict[str, List[str]]:
        """
        使用所有方法选择特征
        
        Args:
            X: 特征矩阵
            y: 目标变量
            top_n: 选择的特征数量
            
        Returns:
            每种方法选择的特征字典
        """
        logger.info(f"Running all feature selection methods with top_n={top_n}...")
        
        selectors = self.get_feature_selectors()
        selected_features = {}
        
        for method_name, selector_func in selectors.items():
            try:
                selected_features[method_name] = selector_func(X, y, top_n)
                self.selected_features[method_name] = selected_features[method_name]
            except Exception as e:
                logger.error(f"Error in {method_name} feature selection: {e}")
                selected_features[method_name] = []
        
        logger.info("All feature selection methods completed")
        return selected_features
    
    def get_common_features(self, min_methods: int = 2) -> List[str]:
        """
        获取被多个方法共同选择的特征
        
        Args:
            min_methods: 最少被选择的方法数量
            
        Returns:
            共同特征列表
        """
        if not self.selected_features:
            logger.warning("No features have been selected yet. Run select_all_features first.")
            return []
        
        # 统计每个特征被选择的方法数量
        feature_counts = {}
        for method, features in self.selected_features.items():
            for feature in features:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        # 筛选被多个方法选择的特征
        common_features = [feature for feature, count in feature_counts.items() if count >= min_methods]
        
        logger.info(f"Found {len(common_features)} features selected by at least {min_methods} methods")
        return common_features
    
    def save_selected_features(self, output_path: str = None) -> str:
        """
        保存选择的特征到JSON文件
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            保存的文件路径
        """
        if not self.selected_features:
            logger.warning("No features have been selected yet.")
            return ""
        
        if output_path is None:
            output_path = f"outputs/selected_features_top{len(list(self.selected_features.values())[0])}.json"
        
        # 确保输出目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(self.selected_features, f, indent=4)
        
        logger.info(f"Selected features saved to: {output_path}")
        return output_path
    
    def load_selected_features(self, file_path: str) -> Dict[str, List[str]]:
        """
        从JSON文件加载选择的特征
        
        Args:
            file_path: 文件路径
            
        Returns:
            选择的特征字典
        """
        with open(file_path, "r") as f:
            self.selected_features = json.load(f)
        
        logger.info(f"Selected features loaded from: {file_path}")
        return self.selected_features
    
    def get_feature_importance_summary(self) -> pd.DataFrame:
        """
        获取特征重要性摘要
        
        Returns:
            特征重要性摘要DataFrame
        """
        if not self.feature_importance_scores:
            logger.warning("No feature importance scores available.")
            return pd.DataFrame()
        
        # 创建特征重要性摘要
        summary_data = []
        for method, scores in self.feature_importance_scores.items():
            for feature, score in scores.items():
                summary_data.append({
                    'Method': method,
                    'Feature': feature,
                    'Importance_Score': score
                })
        
        summary_df = pd.DataFrame(summary_data)
        return summary_df
    
    def plot_feature_importance(self, method: str = None, top_n: int = 10, 
                              save_path: str = None) -> None:
        """
        绘制特征重要性图
        
        Args:
            method: 特征选择方法，如果为None则绘制所有方法
            top_n: 显示前N个特征
            save_path: 保存路径
        """
        import matplotlib.pyplot as plt
        
        if method:
            if method not in self.feature_importance_scores:
                logger.error(f"Method {method} not found in feature importance scores")
                return
            
            scores = self.feature_importance_scores[method]
            top_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            features, importance = zip(*top_features)
            
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(features)), importance)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Importance Score')
            plt.title(f'Top {top_n} Features by {method}')
            plt.gca().invert_yaxis()
            
        else:
            # 绘制所有方法
            fig, axes = plt.subplots(1, len(self.feature_importance_scores), 
                                   figsize=(5*len(self.feature_importance_scores), 6))
            
            if len(self.feature_importance_scores) == 1:
                axes = [axes]
            
            for i, (method, scores) in enumerate(self.feature_importance_scores.items()):
                top_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
                features, importance = zip(*top_features)
                
                axes[i].barh(range(len(features)), importance)
                axes[i].set_yticks(range(len(features)))
                axes[i].set_yticklabels(features)
                axes[i].set_xlabel('Importance Score')
                axes[i].set_title(f'{method} - Top {top_n} Features')
                axes[i].invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to: {save_path}")
        
        plt.show()

def main():
    """主函数，用于测试特征选择功能"""
    from data_loader import DataLoader
    from data_preprocessor import DataPreprocessor
    
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
    
    # 初始化特征选择器
    feature_selector = FeatureSelector()
    
    # 使用所有方法选择特征
    selected_features = feature_selector.select_all_features(X_train_balanced, y_train_balanced, top_n=15)
    
    # 打印结果
    for method, features in selected_features.items():
        print(f"\n{method} selected features:")
        for feature in features:
            print(f"  - {feature}")
    
    # 获取共同特征
    common_features = feature_selector.get_common_features(min_methods=2)
    print(f"\nCommon features selected by at least 2 methods: {common_features}")
    
    # 保存结果
    feature_selector.save_selected_features()
    
    # 绘制特征重要性
    feature_selector.plot_feature_importance(save_path="outputs/feature_importance.png")
    
    return selected_features

if __name__ == "__main__":
    main() 