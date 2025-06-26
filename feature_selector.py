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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureSelector:
    """Feature selector class, providing various feature selection methods"""
    
    def __init__(self):
        """Initializes the feature selector"""
        self.selected_features = {}
        self.feature_importance_scores = {}
        
    def select_features_by_l1(self, X: pd.DataFrame, y: pd.Series, top_n: int = 15) -> List[str]:
        """
        Selects features using L1 regularization
        
        Args:
            X: Feature matrix
            y: Target variable
            top_n: Number of features to select
            
        Returns:
            A list of selected features
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
        Selects features using Mutual Information
        
        Args:
            X: Feature matrix
            y: Target variable
            top_n: Number of features to select
            
        Returns:
            A list of selected features
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
        Selects features using tree-based feature importance
        
        Args:
            X: Feature matrix
            y: Target variable
            top_n: Number of features to select
            
        Returns:
            A list of selected features
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
        Gets all available feature selection methods
        
        Returns:
            A dictionary of feature selection methods
        """
        return {
            'L1': self.select_features_by_l1,
            'MutualInfo': self.select_features_by_mi,
            'TreeImportance': self.select_features_by_tree
        }
    
    def select_features_by_method(self, method: str, X: pd.DataFrame, y: pd.Series, 
                                 top_n: int = 15) -> List[str]:
        """
        Selects features based on a specified method
        
        Args:
            method: The name of the feature selection method
            X: Feature matrix
            y: Target variable
            top_n: Number of features to select
            
        Returns:
            A list of selected features
        """
        selectors = self.get_feature_selectors()
        
        if method not in selectors:
            raise ValueError(f"Unknown feature selection method: {method}. Available methods: {list(selectors.keys())}")
        
        return selectors[method](X, y, top_n)
    
    def select_features_multiple_topn(self, X: pd.DataFrame, y: pd.Series, top_n_list: List[int]) -> Dict[int, Dict[str, List[str]]]:
        """
        Runs all feature selection methods with multiple top_n values
        
        Args:
            X: Feature matrix
            y: Target variable
            top_n_list: A list of top_n values, e.g., [5, 10, 15]
            
        Returns:
            A nested dictionary in the format {top_n: {method: features}}
        """
        logger.info(f"Running feature selection with multiple top_n values: {top_n_list}")
        
        results = {}
        
        for top_n in top_n_list:
            logger.info(f"Processing top_n = {top_n}")
            results[top_n] = self.select_all_features(X, y, top_n)
            
        logger.info("Multiple top_n feature selection completed")
        return results
    
    def select_all_features(self, X: pd.DataFrame, y: pd.Series, top_n: int = 15) -> Dict[str, List[str]]:
        """
        Selects features using all available methods
        
        Args:
            X: Feature matrix
            y: Target variable
            top_n: Number of features to select
            
        Returns:
            A dictionary of features selected by each method
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
        Gets features that are commonly selected by multiple methods
        
        Args:
            min_methods: The minimum number of methods a feature must be selected by
            
        Returns:
            A list of common features
        """
        if not self.selected_features:
            logger.warning("No features have been selected yet. Run select_all_features first.")
            return []
        
        # Count the number of methods that selected each feature
        feature_counts = {}
        for method, features in self.selected_features.items():
            for feature in features:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        # Filter features selected by multiple methods
        common_features = [feature for feature, count in feature_counts.items() if count >= min_methods]
        
        logger.info(f"Found {len(common_features)} features selected by at least {min_methods} methods")
        return common_features
    
    def save_selected_features(self, output_path: str = None) -> str:
        """
        Saves the selected features to a JSON file
        
        Args:
            output_path: The output file path
            
        Returns:
            The path to the saved file
        """
        if not self.selected_features:
            logger.warning("No features have been selected yet.")
            return ""
        
        if output_path is None:
            output_path = f"outputs/selected_features_top{len(list(self.selected_features.values())[0])}.json"
        
        # Ensure the output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(self.selected_features, f, indent=4)
        
        logger.info(f"Selected features saved to: {output_path}")
        return output_path
    
    def load_selected_features(self, file_path: str) -> Dict[str, List[str]]:
        """
        Loads selected features from a JSON file
        
        Args:
            file_path: The file path
            
        Returns:
            A dictionary of selected features
        """
        with open(file_path, "r") as f:
            self.selected_features = json.load(f)
        
        logger.info(f"Selected features loaded from: {file_path}")
        return self.selected_features
    
    def get_feature_importance_summary(self) -> pd.DataFrame:
        """
        Gets feature importance summary
        
        Returns:
            Feature importance summary DataFrame
        """
        if not self.feature_importance_scores:
            logger.warning("No feature importance scores available.")
            return pd.DataFrame()
        
        # Create feature importance summary
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
        Plots feature importance
        
        Args:
            method: Feature selection method, if None plots all methods
            top_n: Display top N features
            save_path: Save path
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
            # No x-ticks for barh
        else:
            # Plot all methods
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

    def display_multiple_topn_results(self, multiple_results: Dict[int, Dict[str, List[str]]]) -> None:
        """
        Displays multiple top_n results in table format

        Args:
            multiple_results: Return result of select_features_multiple_topn
        """
        try:
            from IPython.display import display
            import pandas as pd
        except ImportError:
            logger.warning("IPython or pandas not found. Displaying as plain text.")
            display = print

        # Create detailed result table
        table_data = []
        for top_n, results_for_top_n in multiple_results.items():
            for method, features in results_for_top_n.items():
                table_data.append({
                    'Top N': top_n,
                    'Method': method,
                    'Selected Features': features
                })
        
        results_df = pd.DataFrame(table_data)
        
        print("\n📊 Multiple Top N Value Feature Selection Detailed Results:")
        with pd.option_context('display.max_colwidth', 100):
            display(results_df)
        
        # Create common feature summary table
        common_features_data = []
        methods = list(multiple_results.get(list(multiple_results.keys())[0], {}).keys())
        num_methods = len(methods)

        for top_n, results_for_top_n in multiple_results.items():
            # Temporarily set current result to use get_common_features
            self.selected_features = results_for_top_n
            common_feats_2 = self.get_common_features(min_methods=2)
            common_feats_all = self.get_common_features(min_methods=num_methods)
            
            common_features_data.append({
                'Top N': top_n,
                f'Common Features (>=2 methods)': common_feats_2,
                f'Common Features (all {num_methods} methods)': common_feats_all
            })
        
        common_features_df = pd.DataFrame(common_features_data)
        
        print("\n🔍 Common Features Summary by Top N Value:")
        with pd.option_context('display.max_colwidth', 100):
            display(common_features_df)

    def plot_feature_selection_matrix(self, multiple_results: Dict[int, Dict[str, List[str]]], save_path: str = None) -> None:
        """
        Plots feature selection results in matrix heatmap format

        Args:
            multiple_results: Return result of select_features_multiple_topn
            save_path: Image save path
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
        except ImportError:
            logger.warning("matplotlib or seaborn not found. Skipping plotting.")
            return
            
        print("\n🎨 Generating Feature Selection Matrix Visualization Chart:")
        
        top_n_values = sorted(multiple_results.keys())
        num_top_n = len(top_n_values)

        fig, axes = plt.subplots(1, num_top_n, figsize=(6 * num_top_n, 10), sharey=False)
        if num_top_n == 1:
            axes = [axes]

        fig.suptitle('Feature Selection Matrix by Different Top N Values', fontsize=16, y=1.02)

        for i, top_n in enumerate(top_n_values):
            results_for_top_n = multiple_results[top_n]
            all_selected_features = sorted(list(set(feat for feats in results_for_top_n.values() for feat in feats)))
            selection_matrix = pd.DataFrame(index=all_selected_features)
            for method, features in results_for_top_n.items():
                selection_matrix[method] = [1 if f in features else 0 for f in all_selected_features]
            sns.heatmap(selection_matrix, ax=axes[i], cmap='Blues', cbar=False, annot=True, fmt='.0f')
            axes[i].set_title(f'Top N = {top_n}')
            axes[i].set_xlabel('Method')
            axes[i].set_ylabel('Feature')
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
            axes[i].set_yticklabels(axes[i].get_yticklabels(), rotation=0)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature selection matrix plot saved to: {save_path}")
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
    
    # 使用多个top_n值运行所有特征选择方法
    multiple_results = feature_selector.select_features_multiple_topn(X_train_balanced, y_train_balanced, [5, 10, 15])
    
    # 绘制多个top_n值的结果
    feature_selector.display_multiple_topn_results(multiple_results)

    # 可视化特征选择矩阵
    feature_selector.plot_feature_selection_matrix(multiple_results, save_path='outputs/feature_selection_matrix.png')

    return selected_features

if __name__ == "__main__":
    main() 