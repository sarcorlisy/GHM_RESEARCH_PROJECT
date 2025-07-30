#!/usr/bin/env python3
"""
Simple script to run sensitivity analysis
简单的敏感性分析运行脚本
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from sensitivity_analyzer import SensitivityAnalyzer
from pipeline_config import DATA_PATHS

def main():
    """主函数"""
    print("=" * 60)
    print("Hospital Readmission Prediction - Sensitivity Analysis")
    print("=" * 60)
    
    try:
        # 创建敏感性分析器
        print("Initializing sensitivity analyzer...")
        analyzer = SensitivityAnalyzer()
        
        # 运行敏感性分析
        print("Running sensitivity analysis...")
        results = analyzer.run_full_sensitivity_analysis()
        
        # 显示结果摘要
        print("\n" + "=" * 60)
        print("SENSITIVITY ANALYSIS COMPLETED")
        print("=" * 60)
        
        successful_analyses = sum(1 for r in results['subgroup_results'].values() 
                                if r['status'] == 'success')
        total_analyses = len(results['subgroup_results'])
        
        print(f"Total subgroups analyzed: {total_analyses}")
        print(f"Successful analyses: {successful_analyses}")
        print(f"Failed analyses: {total_analyses - successful_analyses}")
        
        # 显示每个子组的结果
        print("\nSubgroup Results:")
        print("-" * 40)
        for subgroup_name, result in results['subgroup_results'].items():
            status_icon = "✅" if result['status'] == 'success' else "❌"
            print(f"{status_icon} {subgroup_name}: {result['data_size']} patients")
            
            if result['status'] == 'success' and result['test_results'] is not None:
                best_auc = result['test_results']['AUC'].max()
                print(f"   Best AUC: {best_auc:.4f}")
        
        # 显示输出位置
        output_dir = Path(DATA_PATHS['output_dir']) / 'sensitivity_analysis'
        print(f"\nResults saved in: {output_dir}")
        print(f"Summary report: {output_dir / 'sensitivity_analysis_summary.txt'}")
        print(f"Comparison data: {output_dir / 'subgroup_comparison.csv'}")
        
        print("\n" + "=" * 60)
        print("Analysis completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during sensitivity analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 