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
                           roc_auc_score, confusion_matrix, classification_report,
                           precision_recall_curve)
from sklearn.calibration import calibration_curve
import logging
from typing import Dict, List, Tuple, Any
import joblib
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plot_style

from pipeline_config import MODEL_CONFIG, MODELS, FEATURE_CATEGORIES

# Global plotting switch, checks if plotting libraries are available
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    _PLOTTING_ENABLED = True
except ImportError:
    _PLOTTING_ENABLED = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

class ModelTrainer:
    """Model trainer class, responsible for training and evaluating multiple machine learning models"""
    
    def __init__(self, random_state: int = 42):
        """
        Initializes the model trainer
        
        Args:
            random_state: The random seed
        """
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.cv_results = {}
        self.test_results = {}
        
    def get_models(self) -> Dict[str, Any]:
        """
        Gets all available models
        
        Returns:
            A dictionary of models
        """
        return {
            'LogisticRegression': LogisticRegression(solver='liblinear', random_state=self.random_state),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'XGBoost': XGBClassifier(eval_metric='logloss', random_state=self.random_state)
        }
    
    def evaluate_model_with_cv(self, model: Any, X: pd.DataFrame, y: pd.Series, 
                              cv_folds: int = 5) -> Tuple[float, float]:
        """
        Evaluates a model using cross-validation
        
        Args:
            model: The machine learning model
            X: Feature matrix
            y: Target variable
            cv_folds: Number of cross-validation folds
            
        Returns:
            The average AUC and F1 scores
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
                          X_test: pd.DataFrame = None, y_test: pd.Series = None,
                          feature_method: str = None) -> Dict[str, Any]:
        """
        Trains a single model.
        """
        logger.info(f"Training {model_name}...")
        models = self.get_models()
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(models.keys())}")
        model = models[model_name]
        model.fit(X_train, y_train)
        
        # 保存训练好的模型到trained_models字典
        self.trained_models[model_name] = model
        
        auc_cv, f1_cv = self.evaluate_model_with_cv(model, X_train, y_train, 
                                                   cv_folds=MODEL_CONFIG['cv_folds'])
        results = {
            'model_name': model_name,
            'cv_auc': auc_cv,
            'cv_f1': f1_cv
        }
        logger.info(f"{model_name} training completed - CV AUC: {auc_cv:.3f}, CV F1: {f1_cv:.3f}")
        # 新增：如果有测试集，画概率分布图
        if X_test is not None:
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                os.makedirs('outputs', exist_ok=True)
                plt.figure(figsize=(6,4))
                plt.hist(y_pred_proba, bins=50)
                plt.xlabel('Predicted Probability for Positive Class')
                plt.ylabel('Count')
                plt.title(f'Predicted Probability Distribution\n{model_name} {feature_method or ""}')
                plt.tight_layout()
                fname = f'outputs/proba_hist_{model_name}_{feature_method or "default"}.png'
                plt.savefig(fname)
                plt.close()
                logger.info(f"Saved probability histogram to {fname}")
            except Exception as e:
                logger.warning(f"Could not plot probability histogram for {model_name}: {e}")
        return results
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame = None, y_test: pd.Series = None, feature_method: str = None) -> pd.DataFrame:
        """
        Trains all models
        Args:
            X_train: Training set features
            y_train: Training set target variable
            X_test: Test set features (optional, for probability plot)
            y_test: Test set target (optional)
            feature_method: Feature selection method name (for file naming)
        Returns:
            A DataFrame of training results for all models
        """
        logger.info("Training all models...")
        models = self.get_models()
        results = []
        for model_name in models.keys():
            try:
                result = self.train_single_model(model_name, X_train, y_train, X_test, y_test, feature_method)
                results.append(result)
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                results.append({
                    'model_name': model_name,
                    'cv_auc': 0.0,
                    'cv_f1': 0.0
                })
        results_df = pd.DataFrame(results)
        logger.info("All models training completed")
        return results_df
    
    def train_models_for_feature_sets(self, 
                                     feature_sets: Dict[int, Dict[str, List[str]]],
                                     X_train: pd.DataFrame, 
                                     y_train: pd.Series,
                                     X_test: pd.DataFrame = None, y_test: pd.Series = None) -> pd.DataFrame:
        """
        Trains all models for multiple feature sets
        Args:
            feature_sets: A dictionary of feature sets, format: {top_n: {method: [features...]}}
            X_train: The complete training set features
            y_train: The training set target variable
            X_test: The complete test set features (for probability plot)
            y_test: The test set target (optional)
        Returns:
            A DataFrame containing model performance for all scenarios
        """
        logger.info("Starting model training for multiple feature sets...")
        all_results = []
        for top_n, methods in feature_sets.items():
            for method, features in methods.items():
                if not features:
                    logger.warning(f"Skipping training for top_n={top_n}, method={method} due to empty feature list.")
                    continue
                logger.info(f"--- Training for: top_n={top_n}, method={method} ---")
                X_train_subset = X_train[features]
                X_test_subset = X_test[features] if X_test is not None else None
                scenario_results_df = self.train_all_models(X_train_subset, y_train, X_test_subset, y_test, feature_method=method)
                scenario_results_df['top_n'] = top_n
                scenario_results_df['feature_method'] = method
                all_results.append(scenario_results_df)
        if not all_results:
            logger.error("No models were trained. Please check feature sets.")
            return pd.DataFrame()
        final_results_df = pd.concat(all_results, ignore_index=True)
        logger.info("Completed model training for all feature sets.")
        return final_results_df

    

    def display_training_results(self, results_df: pd.DataFrame, metric='cv_auc') -> None:
        """
        Displays multi-scenario training results in tables and pivot tables

        Args:
            results_df: The results from train_models_for_feature_sets
            metric: The core metric to display in the pivot table
        """
        try:
            from IPython.display import display
        except ImportError:
            display = print
            
        print("\n📊 Detailed Model Training Results:")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            display(results_df)

        print(f"\n📈 Performance Summary by '{metric}' (Pivot Table):")
        
        # Create a pivot table
        pivot_table = results_df.pivot_table(
            index=['top_n', 'feature_method'], 
            columns='model_name', 
            values=metric
        )
        
        # Highlight the maximum value in each row
        display(pivot_table.style.highlight_max(axis=1, color='lightgreen'))

    def plot_training_results(self, results_df: pd.DataFrame, metric='cv_auc', save_path: str = None) -> None:
        """
        Visualizes multi-scenario training results

        Args:
            results_df: The results from train_models_for_feature_sets
            metric: The core metric for visualization
            save_path: Path to save the image
        """
        if not _PLOTTING_ENABLED:
            logger.warning("matplotlib or seaborn not found. Skipping plotting.")
            return

        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        print(f"\n🎨 Generating performance visualization chart based on '{metric}':")

        g = sns.catplot(
            data=results_df,
            x='model_name',
            y=metric,
            hue='feature_method',
            col='top_n',
            kind='bar',
            height=6,
            aspect=0.9,
            palette='viridis',
            legend=False
        )
        
        g.fig.suptitle(f'Model Performance Comparison by {metric}', y=1.03, size=16)
        g.set_axis_labels("Model", f"Cross-validated {metric}")
        g.set_titles("Top {col_name} Features")
        g.despine(left=True)
        
        for ax in g.axes.flat:
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.3f}',
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center',
                            xytext=(0, 9),
                            textcoords='offset points',
                            fontsize=9)
            ax.tick_params(axis='x', rotation=45)

        plt.legend(title='Feature Selection Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout(rect=[0, 0, 0.9, 0.96])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Chart saved to {save_path}")
            
        plt.show()

    def plot_performance_vs_top_n(self, results_df: pd.DataFrame, save_path: str = None) -> None:
        """
        Plots model performance as a function of top_n.
        """
        if not _PLOTTING_ENABLED:
            logger.warning("matplotlib or seaborn not found. Skipping plotting.")
            return
            
        print("\n📈 Generating trend chart of model performance vs. number of features:")

        for metric in ['cv_auc', 'cv_f1']:
            plt.figure(figsize=(14, 8))
            
            sns.lineplot(
                data=results_df,
                x='top_n',
                y=metric,
                hue='model_name',
                style='feature_method',
                markers=True,
                dashes=False
            )
            
            plt.title(f'Model Performance ({metric}) vs. Number of Features (top_n)')
            plt.xlabel('Number of Top Features Selected (top_n)')
            plt.ylabel(f'Cross-validated {metric}')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.legend(title='Model & FS Method')
            plt.xticks(rotation=45)
            
            if save_path:
                path_obj = Path(save_path)
                new_path = path_obj.with_name(f"{path_obj.stem}_{metric}{path_obj.suffix}")
                plt.savefig(new_path, dpi=300, bbox_inches='tight')
                logger.info(f"Chart saved to {new_path}")
                
            plt.show()

    def evaluate_final_model(self, best_config: Dict, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Performs a final evaluation on the test set and generates a series of visualizations
        """
        if not _PLOTTING_ENABLED:
            logger.warning("matplotlib or seaborn not found. Skipping plotting.")
            return
        
        # Train the final model
        logger.info("Retraining the best model on the entire training set for final evaluation...")
        model_name = best_config['model_name']
        model = self.get_models()[model_name]
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        print("\n" + "="*60)
        print("🏆 Final Model Evaluation on Test Set 🏆")
        print("="*60)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Create a 3-panel figure for visualizations
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        fig.suptitle(f'Final Evaluation for {model_name}', fontsize=16)
        
        # Plot Confusion Matrix
        self._plot_confusion_matrix(y_test, y_pred, axes[0])
        
        # Plot Precision-Recall Curve
        self._plot_pr_curve(y_test, y_prob, axes[1])
        
        # Plot Calibration Curve
        self._plot_calibration_curve(y_test, y_prob, model_name, axes[2])
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        
    def _plot_confusion_matrix(self, y_true, y_pred, ax):
        """Helper to plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
    def _plot_pr_curve(self, y_true, y_prob, ax):
        """Helper to plot Precision-Recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ax.plot(recall, precision, marker='.')
        ax.set_title('Precision-Recall Curve')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.grid(True)
        
    def _plot_calibration_curve(self, y_true, y_prob, model_name, ax):
        """Helper to plot calibration curve."""
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10)
        ax.plot(mean_predicted_value, fraction_of_positives, "s-", label=model_name)
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        ax.set_title('Calibration Curve')
        ax.set_xlabel('Mean Predicted Value')
        ax.set_ylabel('Fraction of Positives')
        ax.legend()
        ax.grid(True)

    def evaluate_on_test_set(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Evaluates all trained models on the test set
        
        Args:
            X_test: Test set features
            y_test: Test set target variable
            
        Returns:
            A DataFrame of test results for all models
        """
        logger.info("Evaluating models on the test set...")
        
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
        Gets the best performing model based on a specified metric
        
        Args:
            metric: The metric to use for comparison ('auc' or 'f1')
            
        Returns:
            A tuple containing the name of the best model and the model object
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
        Saves all trained models to disk
        
        Args:
            output_dir: The directory to save the models in
        """
        if not self.trained_models:
            logger.warning("No trained models to save.")
            return
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.trained_models.items():
            model_path = f"{output_dir}/{model_name}.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_name} model to {model_path}")
    
    def load_models(self, models_dir: str = 'outputs/models') -> None:
        """
        Loads trained models from disk
        
        Args:
            models_dir: The directory where the models are saved
        """
        models_path = Path(models_dir)
        if not models_path.exists():
            logger.warning(f"Models directory {models_dir} does not exist.")
            return
        
        for model_file in models_path.glob("*.joblib"):
            model_name = model_file.stem
            model = joblib.load(model_file)
            self.trained_models[model_name] = model
            logger.info(f"Loaded {model_name} model from {model_file}")
    
    def generate_model_report(self, output_path: str = 'outputs/model_report.txt') -> None:
        """
        Generates a text report summarizing the model training and evaluation results
        
        Args:
            output_path: Path to save the report
        """
        if not self.test_results:
            logger.warning("No test results available for report generation.")
            return
        
        # Create output directory
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
            
            # Find the best model
            best_auc_model = max(self.test_results.keys(), 
                               key=lambda x: self.test_results[x]['auc'])
            best_f1_model = max(self.test_results.keys(), 
                              key=lambda x: self.test_results[x]['f1'])
            
            f.write(f"\nBest Model by AUC: {best_auc_model} ({self.test_results[best_auc_model]['auc']:.3f})\n")
            f.write(f"Best Model by F1: {best_f1_model} ({self.test_results[best_f1_model]['f1']:.3f})\n")
        
        logger.info(f"Model report generated: {output_path}")
    
    def plot_model_comparison(self, save_path: str = 'outputs/model_comparison.png') -> None:
        """
        Plots a comparison of model performance (CV vs. Test)
        
        Args:
            save_path: Path to save the plot
        """
        if not self.cv_results or not self.test_results:
            logger.warning("CV or test results not available for plotting comparison.")
            return

        if not _PLOTTING_ENABLED:
            logger.warning("matplotlib or seaborn not found. Skipping plotting.")
            return
            
        cv_melt = self.cv_results.melt(id_vars='model_name', var_name='metric', value_name='score')
        cv_melt['type'] = 'Cross-validation'
        
        test_melt = self.test_results.melt(id_vars='model_name', var_name='metric', value_name='score')
        test_melt['type'] = 'Test Set'
        
        combined_results = pd.concat([cv_melt, test_melt])
        
        plt.figure(figsize=(14, 8))
        sns.barplot(data=combined_results, x='metric', y='score', hue='model_name')
        plt.title('Model Performance Comparison: CV vs. Test Set')
        plt.ylabel('Score')
        plt.xlabel('Metric')
        plt.xticks(rotation=45)
        plt.legend(title='Model')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Model comparison plot saved to {save_path}")
            
        plt.show()

    def run_grouped_feature_modeling(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, feature_selectors: dict, top_ns: list, model_names: list):
        """
        For each feature group, FS method, and model, perform modeling and output AUC, F1, and probability distribution histogram for each group.
        Automatically reads pipeline_config.FEATURE_CATEGORIES.
        At the end, automatically visualizes category distribution.
        """
        from pipeline_config import FEATURE_CATEGORIES
        from sklearn.metrics import roc_auc_score, f1_score
        import matplotlib.pyplot as plt
        import os
        import pandas as pd
        os.makedirs('outputs', exist_ok=True)
        results = []
        for fs_name, fs_func in feature_selectors.items():
            for top_n in top_ns:
                selected_features = fs_func(X_train, y_train, top_n=top_n)
                for model_name in model_names:
                    models = self.get_models()
                    if model_name not in models:
                        continue
                    model = models[model_name]
                    X_train_sel = X_train[selected_features]
                    X_test_sel = X_test[selected_features]
                    model.fit(X_train_sel, y_train)
                    y_pred = model.predict(X_test_sel)
                    y_pred_proba = model.predict_proba(X_test_sel)[:, 1]
                    auc = roc_auc_score(y_test, y_pred_proba)
                    f1 = f1_score(y_test, y_pred)
                    # 概率分布图
                    plt.figure(figsize=(6,4))
                    plt.hist(y_pred_proba, bins=50)
                    plt.xlabel('Predicted Probability for Positive Class')
                    plt.ylabel('Count')
                    plt.title(f'Proba: {model_name} | {fs_name} | Top{top_n}')
                    plt.tight_layout()
                    fname = f'outputs/proba_hist_{model_name}_{fs_name}_top{top_n}.png'
                    plt.savefig(fname)
                    plt.close()
                    # 统计各类别特征数量
                    cat_count = {cat: 0 for cat in FEATURE_CATEGORIES if cat != 'Label'}
                    for feat in selected_features:
                        for cat, feats in FEATURE_CATEGORIES.items():
                            if cat == 'Label': continue
                            if feat in feats:
                                cat_count[cat] += 1
                    results.append({'model': model_name, 'fs': fs_name, 'top_n': top_n, 'auc': auc, 'f1': f1, **cat_count})
        results_df = pd.DataFrame(results)
        results_df.to_csv('outputs/grouped_model_results.csv', index=False)
        print('Grouped feature modeling results saved to outputs/grouped_model_results.csv')
        print(results_df)
        # 自动可视化类别分布
        try:
            from result_analyzer import ResultAnalyzer
            analyzer = ResultAnalyzer()
            analyzer.plot_category_distribution(results_df.melt(id_vars=['model','fs','top_n','auc','f1'], var_name='Category', value_name='Count'))
        except Exception as e:
            print(f'Category distribution plot failed: {e}')

    def run_full_grouped_feature_experiment(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        feature_selectors,
        top_ns,
        metric='cv_f1'
    ):
        """
        One-stop grouped feature selection + grouped model training + visualization
        """
        # 1. 自动生成分组特征
        feature_sets = {}
        for top_n in top_ns:
            feature_sets[top_n] = {}
            for method_name, method_func in feature_selectors.items():
                feature_sets[top_n][method_name] = method_func(X_train, y_train, top_n=top_n)

        # 2. 分组训练
        grouped_results_df = self.train_models_for_feature_sets(
            feature_sets=feature_sets,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

        # 3. 可视化
        self.display_training_results(grouped_results_df, metric=metric)
        self.plot_training_results(grouped_results_df, metric=metric)
        self.plot_performance_vs_top_n(grouped_results_df)

        return grouped_results_df

    def run_category_grouped_feature_experiment(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        feature_categories,
        feature_selectors,
        model_names,
        top_n=10,
        metric='cv_f1'
    ):
        """
        For each feature category, perform FS and model training separately, and output the performance of FS*Model for each category.
        """
        results = []
        for cat_name, cat_features in feature_categories.items():
            # 只保留当前cat的特征（且要在X_train中实际存在）
            valid_features = [f for f in cat_features if f in X_train.columns]
            if not valid_features:
                continue
            X_train_cat = X_train[valid_features]
            X_test_cat = X_test[valid_features]
            for fs_name, fs_func in feature_selectors.items():
                selected_features = fs_func(X_train_cat, y_train, top_n=top_n)
                for model_name in model_names:
                    try:
                        res = self.train_single_model(
                            model_name,
                            X_train_cat[selected_features],
                            y_train,
                            X_test_cat[selected_features],
                            y_test,
                            feature_method=f"{cat_name}-{fs_name}"
                        )
                        res['feature_category'] = cat_name
                        res['feature_selector'] = fs_name
                        results.append(res)
                    except Exception as e:
                        print(f"Error in {cat_name}-{fs_name}-{model_name}: {e}")
        results_df = pd.DataFrame(results)
        # 可选：自动可视化
        if not results_df.empty:
            print("\n📊 分类分组FS*Model表现：")
            from IPython.display import display
            display(results_df)
        return results_df

    def plot_cat_fs_model_performance(self, results_df, metric='cv_f1'):
        """
        Grouped bar chart: For each feature category, show the performance of different FS and Model
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(18, 8))
        sns.set(style="whitegrid")
        g = sns.catplot(
            data=results_df,
            x='feature_category',
            y=metric,
            hue='model_name',
            col='feature_selector',
            kind='bar',
            height=6,
            aspect=1.1,
            palette='viridis'
        )
        g.set_titles("FS: {col_name}")
        g.set_axis_labels("Feature Category", metric)
        g.fig.suptitle("Model Performance by Feature Category", y=1.05, fontsize=16)
        for ax in g.axes.flat:
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.3f}',
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center',
                            xytext=(0, 9),
                            textcoords='offset points',
                            fontsize=9)
            ax.tick_params(axis='x', rotation=30)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()

    def plot_all_cat_heatmaps(self, results_df, metric='cv_f1'):
        """
        Merge all category FS*Model heatmaps into one large figure
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        cats = results_df['feature_category'].unique()
        n = len(cats)
        fig, axes = plt.subplots(1, n, figsize=(4*n, 4), constrained_layout=True)
        if n == 1:
            axes = [axes]
        for i, cat in enumerate(cats):
            ax = axes[i]
            pivot = results_df[results_df['feature_category'] == cat].pivot(
                index='feature_selector', columns='model_name', values=metric
            )
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax, cbar=(i==n-1))
            ax.set_title(cat, fontsize=11)
            ax.set_ylabel('FS Method')
            ax.set_xlabel('Model')
        plt.suptitle('Performance Matrix of FS & Model by Feature Category', fontsize=14, y=1.02)
        plt.show()

    def plot_all_cat_heatmaps_multi_topn(self, results_df, metric='cv_f1'):
        """
        When there are multiple top_n, merge all category FS*Model heatmaps into one large figure (one row per top_n, one column per category)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        cats = results_df['feature_category'].unique()
        top_ns = sorted(results_df['top_n'].unique())
        nrow, ncol = len(top_ns), len(cats)
        fig, axes = plt.subplots(nrow, ncol, figsize=(4*ncol, 4*nrow), constrained_layout=True)
        for i, top_n in enumerate(top_ns):
            for j, cat in enumerate(cats):
                ax = axes[i, j] if nrow > 1 else axes[j]
                pivot = results_df[(results_df['feature_category'] == cat) & (results_df['top_n'] == top_n)].pivot(
                    index='feature_selector', columns='model_name', values=metric
                )
                sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax, cbar=(i==0 and j==ncol-1))
                ax.set_title(f"{cat} (TopN={top_n})", fontsize=11)
                ax.set_ylabel('FS Method')
                ax.set_xlabel('Model')
        plt.suptitle('Performance Matrix of FS & Model by Feature Category and TopN', fontsize=14, y=1.02)
        plt.show()

    def evaluate_all_combinations_on_val_and_test(
        self, feature_sets, X_train, y_train, X_val, y_val, X_test, y_test
    ):
        """
        For all combinations of feature selection methods, top_n, and models, output validation and test AUC/F1.
        Args:
            feature_sets: {top_n: {fs_method: [features...]}}
            X_train, y_train: Training set
            X_val, y_val: Validation set
            X_test, y_test: Test set
        Returns:
            DataFrame: val/test AUC, F1 for each combination
        """
        import pandas as pd
        from sklearn.metrics import roc_auc_score, f1_score
        results = []
        for top_n, methods in feature_sets.items():
            for fs_method, features in methods.items():
                for model_name in self.get_models().keys():
                    model = self.get_models()[model_name]
                    model.fit(X_train[features], y_train)
                    # 验证集
                    y_val_pred = model.predict(X_val[features])
                    y_val_prob = model.predict_proba(X_val[features])[:, 1]
                    val_auc = roc_auc_score(y_val, y_val_prob)
                    val_f1 = f1_score(y_val, y_val_pred)
                    # 测试集
                    y_test_pred = model.predict(X_test[features])
                    y_test_prob = model.predict_proba(X_test[features])[:, 1]
                    test_auc = roc_auc_score(y_test, y_test_prob)
                    test_f1 = f1_score(y_test, y_test_pred)
                    results.append({
                        'top_n': top_n,
                        'fs_method': fs_method,
                        'model': model_name,
                        'val_auc': val_auc,
                        'val_f1': val_f1,
                        'test_auc': test_auc,
                        'test_f1': test_f1,
                    })
        return pd.DataFrame(results)

    def grid_search_on_validation(self, fs_func, model_cls, param_grid, top_ns, X_train, y_train, X_val, y_val):
        """
        For the specified feature selection method, model, and top_n, loop through parameter combinations, and output AUC/F1 only on the validation set.
        Args:
            fs_func: Feature selection function
            model_cls: Model class
            param_grid: dict, parameter grid
            top_ns: list, list of top_n
            X_train, y_train: Training set
            X_val, y_val: Validation set
        Returns:
            DataFrame: val_auc/val_f1 for each parameter and top_n
        """
        import itertools
        import pandas as pd
        from sklearn.metrics import roc_auc_score, f1_score
        results = []
        keys, values = zip(*param_grid.items()) if param_grid else ([], [])
        for top_n in top_ns:
            features = fs_func(X_train, y_train, top_n=top_n)
            for param_combo in itertools.product(*values) if values else [()]:
                params = dict(zip(keys, param_combo))
                model = model_cls(**params) if params else model_cls()
                model.fit(X_train[features], y_train)
                y_val_pred = model.predict(X_val[features])
                y_val_prob = model.predict_proba(X_val[features])[:, 1]
                val_auc = roc_auc_score(y_val, y_val_prob)
                val_f1 = f1_score(y_val, y_val_pred)
                results.append({
                    'top_n': top_n,
                    **params,
                    'val_auc': val_auc,
                    'val_f1': val_f1
                })
        return pd.DataFrame(results)

    def evaluate_on_test_with_config(self, fs_func, model_cls, params, top_n, X_train, y_train, X_test, y_test):
        """
        For the specified feature selection method, model, top_n, and parameters, output AUC/F1 on the test set.
        Args:
            fs_func: Feature selection function
            model_cls: Model class
            params: dict, model parameters
            top_n: int
            X_train, y_train: Training set
            X_test, y_test: Test set
        Returns:
            dict: test_auc, test_f1
        """
        from sklearn.metrics import roc_auc_score, f1_score
        features = fs_func(X_train, y_train, top_n=top_n)
        model = model_cls(**params) if params else model_cls()
        model.fit(X_train[features], y_train)
        y_test_pred = model.predict(X_test[features])
        y_test_prob = model.predict_proba(X_test[features])[:, 1]
        test_auc = roc_auc_score(y_test, y_test_prob)
        test_f1 = f1_score(y_test, y_test_pred)
        return {'test_auc': test_auc, 'test_f1': test_f1}

    def param_search_on_fixed_features(self, feature_list, model_cls, param_grid, X_train, y_train, X_val, y_val):
        """
        Loop through parameter combinations only on the selected feature subset, outputting validation AUC/F1.
        Args:
            feature_list: list, selected feature names
            model_cls: Model class
            param_grid: dict, parameter grid
            X_train, y_train: Training set
            X_val, y_val: Validation set
        Returns:
            DataFrame: val_auc/val_f1 for each parameter
        """
        import itertools
        import pandas as pd
        from sklearn.metrics import roc_auc_score, f1_score
        results = []
        keys, values = zip(*param_grid.items()) if param_grid else ([], [])
        for param_combo in itertools.product(*values) if values else [()]:
            params = dict(zip(keys, param_combo))
            model = model_cls(**params) if params else model_cls()
            model.fit(X_train[feature_list], y_train)
            y_val_pred = model.predict(X_val[feature_list])
            y_val_prob = model.predict_proba(X_val[feature_list])[:, 1]
            val_auc = roc_auc_score(y_val, y_val_prob)
            val_f1 = f1_score(y_val, y_val_pred)
            results.append({
                **params,
                'val_auc': val_auc,
                'val_f1': val_f1
            })
        return pd.DataFrame(results)

    def param_search_all_models_on_fixed_features(self, feature_list, model_classes, param_grids, X_train, y_train, X_val, y_val):
        """
        For the same feature subset, loop through all models, automatically tune parameters for each model, and output the best parameters and scores.
        Args:
            feature_list: list, selected feature names
            model_classes: dict, {model_name: model_cls}
            param_grids: dict, {model_name: param_grid}
            X_train, y_train: Training set
            X_val, y_val: Validation set
        Returns:
            DataFrame: best parameters and val_auc/val_f1 for each model
        """
        import pandas as pd
        results = []
        for model_name, model_cls in model_classes.items():
            param_grid = param_grids[model_name]
            val_results = self.param_search_on_fixed_features(
                feature_list, model_cls, param_grid, X_train, y_train, X_val, y_val
            )
            best_row = val_results.loc[val_results['val_auc'].idxmax()]
            result = {'model': model_name, 'val_auc': best_row['val_auc'], 'val_f1': best_row['val_f1']}
            for k in param_grid.keys():
                result[k] = best_row[k]
            results.append(result)
        return pd.DataFrame(results)

    def batch_param_search_and_test(
        self, fs_name, top_ns, selected_features_dict, model_classes, param_grids,
        X_train_balanced, y_train_balanced, X_val, y_val, X_test, y_test,
        cv_folds: int = None  # New: cv_folds parameter
    ):
        """
        Batch parameter tuning and test evaluation for all models for the specified FS and top_ns, returning all validation and test optimal result DataFrames.
        Supports cross-validation mode.
        
        Args:
            cv_folds: If set, use cross-validation; if None, use holdout validation
        """
        import pandas as pd
        from sklearn.model_selection import GridSearchCV
        
        all_val_results = []
        all_test_results = []
        
        for top_n in top_ns:
            feature_list = selected_features_dict[(top_n, fs_name)]
            
            for model_name, model_cls in model_classes.items():
                param_grid = param_grids[model_name]
                
                if cv_folds is not None:
                    # 使用交叉验证模式
                    print(f"Running GridSearchCV with cv={cv_folds} for {model_name}...")
                    
                    # 创建带random_state的模型实例
                    if model_name == 'RandomForest':
                        base_model = model_cls(random_state=self.random_state)
                    elif model_name == 'LogisticRegression':
                        base_model = model_cls(random_state=self.random_state)
                    elif model_name == 'XGBoost':
                        base_model = model_cls(random_state=self.random_state)
                    else:
                        base_model = model_cls()
                    
                    grid = GridSearchCV(
                        estimator=base_model,
                        param_grid=param_grid,
                        scoring='roc_auc',
                        cv=cv_folds,
                        n_jobs=-1,
                        return_train_score=False
                    )
                    
                    # 在训练集上进行交叉验证
                    grid.fit(X_train_balanced[feature_list], y_train_balanced)
                    best_params = grid.best_params_
                    cv_score = grid.best_score_
                    
                    # 验证集评估
                    y_val_pred = grid.best_estimator_.predict(X_val[feature_list])
                    y_val_prob = grid.best_estimator_.predict_proba(X_val[feature_list])[:, 1]
                    val_auc = roc_auc_score(y_val, y_val_prob)
                    val_f1 = f1_score(y_val, y_val_pred)
                    
                    # 测试集评估
                    y_test_pred = grid.best_estimator_.predict(X_test[feature_list])
                    y_test_prob = grid.best_estimator_.predict_proba(X_test[feature_list])[:, 1]
                    test_auc = roc_auc_score(y_test, y_test_prob)
                    test_f1 = f1_score(y_test, y_test_pred)
                    
                    # 保存验证结果
                    val_result = {
                        'model': model_name,
                        'fs': fs_name,
                        'top_n': top_n,
                        **best_params,
                        'val_auc': val_auc,
                        'val_f1': val_f1,
                        'cv_score': cv_score
                    }
                    all_val_results.append(val_result)
                    
                    # 保存测试结果
                    test_result = {
                        'model': model_name,
                        'fs': fs_name,
                        'top_n': top_n,
                        **best_params,
                        'test_auc': test_auc,
                        'test_f1': test_f1,
                        'cv_score': cv_score
                    }
                    all_test_results.append(test_result)
                    
                else:
                    # 使用原有的holdout验证模式
                    # 1. validation调参
                    val_results = self.param_search_on_fixed_features(
                        feature_list, model_cls, param_grid,
                        X_train_balanced, y_train_balanced, X_val, y_val
                    )
                    val_results['model'] = model_name
                    val_results['fs'] = fs_name
                    val_results['top_n'] = top_n
                    all_val_results.append(val_results)

                    # 2. 选最优参数
                    best_row = val_results.loc[val_results['val_auc'].idxmax()]
                    param_keys = list(param_grid.keys())
                    params = {k: best_row[k] for k in param_keys}

                    # 3. test集评估
                    test_result = self.evaluate_on_test_with_config(
                        lambda X, y, top_n: feature_list,
                        model_cls, params, top_n,
                        X_train_balanced, y_train_balanced, X_test, y_test
                    )
                    test_result.update({
                        'model': model_name,
                        'fs': fs_name,
                        'top_n': top_n,
                        **params,
                        'val_auc': best_row['val_auc'],
                        'val_f1': best_row['val_f1']
                    })
                    all_test_results.append(test_result)
        
        all_val_results_df = pd.concat(all_val_results, ignore_index=True)
        all_test_results_df = pd.DataFrame(all_test_results)
        return all_val_results_df, all_test_results_df

    def run_grid_search_for_all_fs_methods(
        self, 
        fs_names: List[str], 
        top_n: int, 
        selected_features_dict: Dict, 
        model_classes: Dict, 
        param_grids: Dict, 
        X_train_balanced: pd.DataFrame, 
        y_train_balanced: pd.Series, 
        X_val: pd.DataFrame, 
        y_val: pd.Series, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        cv_folds: int = 3
    ) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame], Dict[Tuple[str, str], Any]]:
        """
        Run GridSearchCV for all feature selection methods
        
        Args:
            fs_names: List of feature selection method names
            top_n: Number of features
            selected_features_dict: Feature selection result dictionary
            model_classes: Model class dictionary
            param_grids: Parameter grid dictionary
            X_train_balanced: Balanced training set features
            y_train_balanced: Balanced training set labels
            X_val: Validation set features
            y_val: Validation set labels
            X_test: Test set features
            y_test: Test set labels
            cv_folds: Number of cross-validation folds
            
        Returns:
            Tuple of (all_val_results, all_test_results, all_cv_results_list, best_models)
        """
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import roc_auc_score, f1_score
        import joblib
        
        all_val_results = []
        all_test_results = []
        all_cv_results_list = []
        best_models = {}  # New: Save all best model objects
        
        for fs_name in fs_names:
            feature_list = selected_features_dict[(top_n, fs_name)]
            val_results = []
            test_results = []
            cv_results_list = []
            
            for model_name, model_cls in model_classes.items():
                print(f"\nGrid search for {fs_name} - {model_name} ...")
                
                # 创建带random_state的模型实例
                if model_name == 'RandomForest':
                    base_model = model_cls(random_state=self.random_state)
                elif model_name == 'LogisticRegression':
                    base_model = model_cls(random_state=self.random_state)
                elif model_name == 'XGBoost':
                    base_model = model_cls(random_state=self.random_state)
                else:
                    base_model = model_cls()
                
                grid = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_grids[model_name],
                    scoring='roc_auc',
                    cv=cv_folds,
                    n_jobs=-1,
                    return_train_score=False
                )
                
                grid.fit(X_train_balanced[feature_list], y_train_balanced)
                best_params = grid.best_params_
                
                # New: Save best model object to dictionary
                best_models[(fs_name, model_name)] = grid.best_estimator_
                # New: Save best model to disk
                model_filename = f'best_model_{fs_name}_{model_name}.pkl'
                joblib.dump(grid.best_estimator_, model_filename)
                print(f"Best model saved to {model_filename}")
                
                # Save all CV results
                all_cv_results = pd.DataFrame(grid.cv_results_)
                all_cv_results['model'] = model_name
                all_cv_results['fs'] = fs_name
                all_cv_results['top_n'] = top_n
                cv_results_list.append(all_cv_results)
                
                # 验证集评估
                y_val_pred = grid.best_estimator_.predict(X_val[feature_list])
                y_val_prob = grid.best_estimator_.predict_proba(X_val[feature_list])[:, 1]
                val_auc = roc_auc_score(y_val, y_val_prob)
                val_f1 = f1_score(y_val, y_val_pred)
                # New: Save n_iter
                n_iter = None
                if hasattr(grid.best_estimator_, 'n_iter_'):
                    n_iter = grid.best_estimator_.n_iter_
                    # Compatible with multi-class cases
                    if hasattr(n_iter, '__len__') and not isinstance(n_iter, str):
                        n_iter = n_iter[0]
                val_results.append({
                    'model': model_name,
                    'fs': fs_name,
                    'top_n': top_n,
                    **best_params,
                    'val_auc': val_auc,
                    'val_f1': val_f1,
                    'n_iter': n_iter
                })
                
                # 测试集评估
                y_test_pred = grid.best_estimator_.predict(X_test[feature_list])
                y_test_prob = grid.best_estimator_.predict_proba(X_test[feature_list])[:, 1]
                test_auc = roc_auc_score(y_test, y_test_prob)
                test_f1 = f1_score(y_test, y_test_pred)
                # New: Save n_iter
                test_results.append({
                    'model': model_name,
                    'fs': fs_name,
                    'top_n': top_n,
                    **best_params,
                    'test_auc': test_auc,
                    'test_f1': test_f1,
                    'n_iter': n_iter
                })
            
            # Convert to DataFrame
            val_df = pd.DataFrame(val_results)
            test_df = pd.DataFrame(test_results)
            all_cv_results_df = pd.concat(cv_results_list, ignore_index=True)
            
            # Add to total result list
            all_val_results.append(val_df)
            all_test_results.append(test_df)
            all_cv_results_list.append(all_cv_results_df)
            
            print(f"{fs_name} GridSearchCV completed")
        
        # New: Return best_models
        return all_val_results, all_test_results, all_cv_results_list, best_models
        

    def compare_parameter_settings(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                 X_test: pd.DataFrame, y_test: pd.Series,
                                 default_results: pd.DataFrame,
                                 tuned_results: pd.DataFrame) -> None:
        """
        Compares model performance between default and tuned parameters
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            default_results: Results from default parameters
            tuned_results: Results from tuned parameters
        """
        print("=" * 60)
        print("PARAMETER SETTING COMPARISON")
        print("=" * 60)
        
        # Merge results for comparison
        comparison_data = []
        
        # Add default parameter results
        for _, row in default_results.iterrows():
            comparison_data.append({
                'Model': row['model_name'],
                'FS_Method': row.get('feature_method', 'default'),
                'Top_N': row.get('top_n', 'default'),
                'Parameter_Setting': 'Default',
                'CV_AUC': row.get('cv_auc', 0),
                'CV_F1': row.get('cv_f1', 0),
                'Test_AUC': row.get('test_auc', 0),
                'Test_F1': row.get('test_f1', 0)
            })
        
        # Add tuned parameter results
        for _, row in tuned_results.iterrows():
            comparison_data.append({
                'Model': row['model_name'],
                'FS_Method': row.get('feature_method', 'default'),
                'Top_N': row.get('top_n', 'default'),
                'Parameter_Setting': 'Tuned',
                'CV_AUC': row.get('cv_auc', 0),
                'CV_F1': row.get('cv_f1', 0),
                'Test_AUC': row.get('test_auc', 0),
                'Test_F1': row.get('test_f1', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display comparison results
        print("\n📊 Performance Comparison:")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            display(comparison_df)
        
        # Calculate performance changes
        print("\n📈 Performance Changes (Tuned - Default):")
        for metric in ['CV_AUC', 'CV_F1', 'Test_AUC', 'Test_F1']:
            default_avg = comparison_df[comparison_df['Parameter_Setting'] == 'Default'][metric].mean()
            tuned_avg = comparison_df[comparison_df['Parameter_Setting'] == 'Tuned'][metric].mean()
            change = tuned_avg - default_avg
            change_pct = (change / default_avg) * 100 if default_avg != 0 else 0
            
            print(f"{metric}: {change:+.3f} ({change_pct:+.1f}%)")
        
        # Visualize comparison
        if _PLOTTING_ENABLED:
            self._plot_parameter_comparison(comparison_df)
        
        return comparison_df

    def _plot_parameter_comparison(self, comparison_df: pd.DataFrame) -> None:
        """Plot parameter comparison results"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        metrics = ['CV_AUC', 'CV_F1', 'Test_AUC', 'Test_F1']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            
            # Create comparison plot
            pivot_data = comparison_df.pivot_table(
                index=['Model', 'FS_Method', 'Top_N'],
                columns='Parameter_Setting',
                values=metric,
                aggfunc='mean'
            )
            
            # Calculate differences
            if 'Default' in pivot_data.columns and 'Tuned' in pivot_data.columns:
                pivot_data['Difference'] = pivot_data['Tuned'] - pivot_data['Default']
                
                # Plot difference distribution
                pivot_data['Difference'].hist(ax=ax, bins=20, alpha=0.7)
                ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
                ax.set_title(f'{metric} Difference Distribution')
                ax.set_xlabel('Tuned - Default')
                ax.set_ylabel('Frequency')
                
                # Add statistics
                mean_diff = pivot_data['Difference'].mean()
                ax.text(0.05, 0.95, f'Mean: {mean_diff:.3f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function to demonstrate the ModelTrainer class"""
    
    # This is a placeholder for demonstration. 
    # In a real pipeline, data would be passed from a previous step.
    try:
        X_train = pd.read_csv('outputs/X_train_scaled.csv')
        y_train = pd.read_csv('outputs/y_train.csv').squeeze()
        X_test = pd.read_csv('outputs/X_test_scaled.csv')
        y_test = pd.read_csv('outputs/y_test.csv').squeeze()
    except FileNotFoundError:
        logger.error("Demo files not found. Please run the data preprocessing step first.")
        return

    trainer = ModelTrainer()
    
    # Train all models and get CV results
    cv_results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    print("\nCV Results:")
    print(cv_results)
    
    # Evaluate on test set
    test_results = trainer.evaluate_on_test_set(X_test, y_test)
    print("\nTest Set Results:")
    print(test_results)

    # Generate and save report
    trainer.generate_model_report()

    # Plot comparison
    trainer.plot_model_comparison()

    # Save models
    trainer.save_models()
    
    # Get and show best model
    best_model_name, best_model = trainer.get_best_model()
    print(f"\nBest model is: {best_model_name}")

if __name__ == "__main__":
    main() 