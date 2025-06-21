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
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    def train_single_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        训练单个模型。
        """
        logger.info(f"Training {model_name}...")
        
        models = self.get_models()
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(models.keys())}")
        
        model = models[model_name]
        
        # 直接在传入的数据上训练
        model.fit(X_train, y_train)
        
        # 在同样的数据上进行交叉验证
        auc_cv, f1_cv = self.evaluate_model_with_cv(model, X_train, y_train, 
                                                   cv_folds=MODEL_CONFIG['cv_folds'])
        
        results = {
            'model_name': model_name,
            'cv_auc': auc_cv,
            'cv_f1': f1_cv
        }
        
        logger.info(f"{model_name} training completed - CV AUC: {auc_cv:.3f}, CV F1: {f1_cv:.3f}")
        return results
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        """
        训练所有模型
        
        Args:
            X_train: 训练集特征
            y_train: 训练集目标变量
            
        Returns:
            所有模型的训练结果DataFrame
        """
        logger.info("Training all models...")
        
        models = self.get_models()
        results = []
        
        for model_name in models.keys():
            try:
                result = self.train_single_model(model_name, X_train, y_train)
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
                                     y_train: pd.Series) -> pd.DataFrame:
        """
        为多个特征集训练所有模型

        Args:
            feature_sets: 特征集字典, 格式为 {top_n: {method: [features...]}}
            X_train: 完整的训练集特征
            y_train: 训练集目标变量

        Returns:
            一个包含所有场景下模型性能的DataFrame
        """
        logger.info("Starting model training for multiple feature sets...")
        all_results = []

        for top_n, methods in feature_sets.items():
            for method, features in methods.items():
                if not features:
                    logger.warning(f"Skipping training for top_n={top_n}, method={method} due to empty feature list.")
                    continue

                logger.info(f"--- Training for: top_n={top_n}, method={method} ---")
                
                # 选取当前场景的特征子集
                X_train_subset = X_train[features]
                
                # 训练所有模型
                scenario_results_df = self.train_all_models(X_train_subset, y_train)
                
                # 添加场景信息
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
        以表格和透视表的形式展示多场景训练结果

        Args:
            results_df: 来自 train_models_for_feature_sets 的结果
            metric: 用于在透视表中展示的核心指标
        """
        try:
            from IPython.display import display
        except ImportError:
            display = print
            
        print("\n📊 详细模型训练结果:")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            display(results_df)

        print(f"\n📈 按'{metric}'指标表现的性能总结 (透视表):")
        
        # 创建透视表
        pivot_table = results_df.pivot_table(
            index=['top_n', 'feature_method'], 
            columns='model_name', 
            values=metric
        )
        
        # 高亮每行的最大值
        display(pivot_table.style.highlight_max(axis=1, color='lightgreen'))

    def plot_training_results(self, results_df: pd.DataFrame, metric='cv_auc', save_path: str = None) -> None:
        """
        可视化多场景训练结果

        Args:
            results_df: 来自 train_models_for_feature_sets 的结果
            metric: 用于可视化的核心指标
            save_path: 图片保存路径
        """
        try:
            # 设置中文字体，以防万一
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
        except ImportError:
            logger.warning("matplotlib or seaborn not found. Skipping plotting.")
            return

        print(f"\n🎨 生成基于'{metric}'指标的性能可视化图表:")

        # 使用catplot可以轻松创建按top_n分组的条形图
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

        # 调整图表细节
        g.fig.suptitle(f'各模型在不同Top N和特征选择方法下的性能 ({metric})', y=1.03, size=16)
        g.set_axis_labels("机器学习模型", f"性能得分 ({metric})")
        g.set_titles("Top N = {col_name}")
        g.despine(left=True)

        # 为每个子图添加数值标签
        for ax in g.axes.flat:
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.3f}',
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center',
                            xytext=(0, 9),
                            textcoords='offset points',
                            fontsize=9)
            ax.tick_params(axis='x', rotation=30)

        # 添加图例
        plt.legend(title='特征选择方法', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout(rect=[0, 0, 0.9, 0.96])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training results plot saved to: {save_path}")

        plt.show()

    def plot_performance_vs_top_n(self, results_df: pd.DataFrame, save_path: str = None) -> None:
        """
        绘制模型性能随 top_n 变化的曲线图。

        Args:
            results_df: 来自 train_models_for_feature_sets 的结果。
            save_path: 可选的图片保存路径。
        """
        if not _PLOTTING_ENABLED: return
        print("\n📈 生成模型性能随特征数量变化的趋势图:")

        # 为两个核心指标（AUC 和 F1）分别绘图
        for metric in ['cv_auc', 'cv_f1']:
            plt.figure(figsize=(14, 8))
            
            sns.lineplot(
                data=results_df,
                x='top_n',
                y=metric,
                hue='model_name',
                style='feature_method',
                marker='o',
                markersize=8,
                palette='tab10'
            )
            
            plt.title(f'模型性能 ({metric}) vs. 特征数量 (Top N)', fontsize=16)
            plt.xlabel('选择的特征数量 (Top N)', fontsize=12)
            plt.ylabel(f'交叉验证得分 ({metric})', fontsize=12)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.legend(title='模型/特征选择方法', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout(rect=[0, 0, 0.85, 1])

            if save_path:
                # 为每个指标保存不同的文件名
                metric_save_path = save_path.replace('.png', f'_{metric}.png')
                plt.savefig(metric_save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Performance vs. top_n plot saved to: {metric_save_path}")

            plt.show()

    def evaluate_final_model(self, best_config: Dict, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        对最终选定的最佳模型配置进行全面的健壮性评估。

        Args:
            best_config: 包含最佳 top_n, feature_method, model_name 的字典。
            X_train, y_train: 完整的训练数据。
            X_test, y_test: 独立的测试数据。
        """
        if not _PLOTTING_ENABLED: return

        print("\n" + "="*60)
        print("          FINAL MODEL ROBUSTNESS EVALUATION          ")
        print("="*60)
        
        # 1. 在完整的训练集上重新训练最佳模型
        model_name = best_config['model_name']
        logger.info(f"Retraining the final best model: {model_name}...")
        model = self.get_models()[model_name]
        model.fit(X_train, y_train)
        
        # 2. 在测试集上进行预测
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # 3. 打印核心指标
        print("\n📊 独立测试集上的最终性能:")
        print(classification_report(y_test, y_pred, target_names=['Not Readmitted', 'Readmitted']))

        # 4. 绘制三种核心评估图
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        fig.suptitle(f'对最佳模型 ({model_name}) 的最终评估', fontsize=18)

        self._plot_confusion_matrix(y_test, y_pred, ax=axes[0])
        self._plot_pr_curve(y_test, y_prob, ax=axes[1])
        self._plot_calibration_curve(y_test, y_prob, model_name, ax=axes[2])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def _plot_confusion_matrix(self, y_true, y_pred, ax):
        """Helper to plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=['Predicted Not Readmitted', 'Predicted Readmitted'],
                    yticklabels=['Actual Not Readmitted', 'Actual Readmitted'])
        ax.set_title('混淆矩阵 (Confusion Matrix)', fontsize=14)
        ax.set_xlabel('预测标签')
        ax.set_ylabel('真实标签')

    def _plot_pr_curve(self, y_true, y_prob, ax):
        """Helper to plot Precision-Recall curve."""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap_score = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, marker='.', label=f'AP = {ap_score:.3f}')
        ax.set_title('精确率-召回率曲线 (PR Curve)', fontsize=14)
        ax.set_xlabel('召回率 (Recall)')
        ax.set_ylabel('精确率 (Precision)')
        ax.grid(True, linestyle='--')
        ax.legend()

    def _plot_calibration_curve(self, y_true, y_prob, model_name, ax):
        """Helper to plot calibration curve."""
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
        ax.plot(prob_pred, prob_true, marker='o', label=model_name)
        ax.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
        ax.set_title('校准曲线 (Calibration Curve)', fontsize=14)
        ax.set_xlabel('预测概率的平均值 (Mean Predicted Probability)')
        ax.set_ylabel('正例的比例 (Fraction of Positives)')
        ax.grid(True, linestyle='--')
        ax.legend()
    
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
    
    # 初始化特征选择器
    feature_selector = FeatureSelector()
    
    # 使用多个top_n值运行所有特征选择方法
    multiple_feature_sets = feature_selector.select_features_multiple_topn(
        X_train_balanced, y_train_balanced, [5, 10, 15]
    )
    
    # 显示特征选择结果
    feature_selector.display_multiple_topn_results(multiple_feature_sets)

    # 初始化模型训练器
    model_trainer = ModelTrainer()

    # 为多个特征集训练模型
    training_results = model_trainer.train_models_for_feature_sets(
        multiple_feature_sets,
        X_train_balanced,
        y_train_balanced
    )
    
    # 显示训练结果
    model_trainer.display_training_results(training_results, metric='cv_f1')
    
    # 可视化训练结果
    model_trainer.plot_training_results(training_results, metric='cv_f1', save_path='outputs/training_performance_comparison.png')
    
    # 在测试集上评估
    test_results = model_trainer.evaluate_on_test_set(X_test, y_test)
    
    print("\nTest Results:")
    print(test_results)
    
    # 获取最佳模型
    best_model_name, best_model = model_trainer.get_best_model('auc')
    
    # 保存模型和生成报告
    model_trainer.save_models()
    model_trainer.generate_model_report()
    model_trainer.plot_model_comparison()
    
    # 对最佳模型进行全面的健壮性评估
    model_trainer.evaluate_final_model(best_config, X_train_balanced, y_train_balanced, X_test, y_test)
    
    return model_trainer, test_results

if __name__ == "__main__":
    main() 