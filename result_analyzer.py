"""
Result Analysis Module for the Hospital Readmission Pipeline.

This module provides tools to analyze and visualize the results from
feature selection and model training stages.
"""
import pandas as pd
import logging
from typing import Dict, List
from pipeline_config import FEATURE_CATEGORIES

# Seaborn and Matplotlib are optional and imported within functions
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    from IPython.display import display
    _PLOTTING_ENABLED = True
except ImportError:
    _PLOTTING_ENABLED = False

logger = logging.getLogger(__name__)

class ResultAnalyzer:
    """
    Analyzes results from the pipeline, focusing on feature categories.
    """
    def __init__(self):
        """Initializes the ResultAnalyzer."""
        self.feature_to_category = self._create_feature_category_map()

    def _create_feature_category_map(self) -> Dict[str, str]:
        """Creates a reverse map from feature name to its main category."""
        mapping = {}
        for category, features in FEATURE_CATEGORIES.items():
            for feature in features:
                mapping[feature] = category
        return mapping

    def _get_feature_category(self, feature_name: str) -> str:
        """
        Gets the main category for a given feature name, handling one-hot encoded features.
        """
        # Direct match
        if feature_name in self.feature_to_category:
            return self.feature_to_category[feature_name]
        
        # Handle one-hot encoded features by checking the prefix
        for base_feature in self.feature_to_category:
            if feature_name.startswith(base_feature + '_'):
                return self.feature_to_category[base_feature]
                
        return 'Unknown'

    def analyze_feature_categories(self, multiple_results: Dict[int, Dict[str, List[str]]]):
        """
        Performs a full analysis of selected features based on their categories.

        Args:
            multiple_results: The results from FeatureSelector's multi-topn run.
        """
        print("="*60)
        print("          FEATURE CATEGORY ANALYSIS          ")
        print("="*60)

        # 1. Create the detailed mapping table
        analysis_data = []
        for top_n, methods in multiple_results.items():
            for method, features in methods.items():
                for feature in features:
                    analysis_data.append({
                        'Top N': top_n,
                        'Method': method,
                        'Feature': feature,
                        'Category': self._get_feature_category(feature)
                    })
        
        if not analysis_data:
            logger.warning("No features were selected, skipping category analysis.")
            return

        analysis_df = pd.DataFrame(analysis_data)
        print("\n📊 Detailed Feature Category Mapping Table:")
        with pd.option_context('display.max_rows', 20):
            display(analysis_df)

        # 2. Create the statistical summary
        stats_df = analysis_df.groupby(['Top N', 'Method', 'Category']).size().reset_index(name='Count')
        
        print("\n📈 Statistical Distribution of Feature Categories per Scenario:")
        pivot_stats = stats_df.pivot_table(index=['Top N', 'Method'], columns='Category', values='Count', fill_value=0)
        display(pivot_stats)

        # 3. Create the visualization
        if _PLOTTING_ENABLED:
            self.plot_category_distribution(stats_df)
        else:
            logger.warning("Plotting is disabled. Please install `seaborn` and `matplotlib`.")

    def plot_category_distribution(self, stats_df: pd.DataFrame, save_path: str = None):
        """
        Visualizes the distribution of feature categories for each scenario.
        Args:
            stats_df: The statistical summary DataFrame.
            save_path: Optional path to save the plot.
        """
        print("\n🎨 Generating Feature Category Distribution Visualization:")
        pivot_df = stats_df.pivot_table(
            index=['Top N', 'Method'], 
            columns='Category', 
            values='Count'
        ).fillna(0)
        top_n_values = sorted(stats_df['Top N'].unique())
        # 横向并排
        n = len(top_n_values)
        fig, axes = plt.subplots(
            1, n,  # 横向排列
            figsize=(7 * n, 6),
            sharey=True
        )
        if n == 1:
            axes = [axes]
        fig.suptitle('Category Distribution for Each Feature Selection Method by Top N', fontsize=16, y=1.02)
        for i, top_n in enumerate(top_n_values):
            ax = axes[i]
            data_subset = pivot_df.loc[top_n]
            data_subset.plot(kind='bar', stacked=True, ax=ax, colormap='viridis', width=0.8)
            ax.set_title(f'Top N = {top_n}')
            ax.set_ylabel('Number of Features')
            ax.tick_params(axis='x', rotation=45)
            if i == n - 1:
                ax.legend(title='Feature Category', bbox_to_anchor=(1.02, 1), loc='upper left')
            else:
                ax.get_legend().remove()
        plt.xlabel('Feature Selection Method')
        plt.tight_layout(rect=[0, 0, 0.92, 0.96])
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Category distribution plot saved to: {save_path}")
        plt.show() 