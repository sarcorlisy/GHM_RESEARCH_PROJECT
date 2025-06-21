"""
Hospital Readmission Prediction Pipeline - 使用示例
"""
import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from main_pipeline import HospitalReadmissionPipeline
import logging

def main():
    """运行pipeline示例"""
    
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("🏥 Hospital Readmission Prediction Pipeline")
    print("=" * 50)
    
    try:
        # 初始化pipeline
        logger.info("Initializing pipeline...")
        pipeline = HospitalReadmissionPipeline()
        
        # 运行完整pipeline
        logger.info("Starting full pipeline execution...")
        results = pipeline.run_full_pipeline()
        
        # 显示结果摘要
        print("\n✅ Pipeline completed successfully!")
        print("\n📊 Results Summary:")
        print("-" * 30)
        
        best_model_name, best_model = results['best_model']
        best_auc = results['test_results'].loc[
            results['test_results']['model_name'] == best_model_name, 'auc'
        ].iloc[0]
        
        print(f"Best Model: {best_model_name}")
        print(f"Best AUC Score: {best_auc:.3f}")
        print(f"Total Features Selected: {len(results['selected_features']['MutualInfo'])}")
        print(f"Training Samples: {len(results['training_results'])} models trained")
        
        # 显示所有模型性能
        print("\n📈 Model Performance Comparison:")
        print("-" * 40)
        print(results['test_results'][['model_name', 'auc', 'f1', 'accuracy']].to_string(index=False))
        
        # 显示选择的特征
        print(f"\n🔍 Top Features Selected by Mutual Information:")
        print("-" * 50)
        for i, feature in enumerate(results['selected_features']['MutualInfo'][:10], 1):
            print(f"{i:2d}. {feature}")
        
        print(f"\n📁 Output files saved to 'outputs/' directory")
        print("📄 Check 'outputs/final_pipeline_report.txt' for detailed report")
        print("📊 Check 'outputs/model_comparison.png' for visualization")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Data file not found - {e}")
        print("Please ensure all required data files are in the project directory:")
        print("  - diabetic_data.csv")
        print("  - IDS_mapping.csv")
        print("  - ccs_icd9_mapping.csv")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Pipeline failed: {e}", exc_info=True)

if __name__ == "__main__":
    main() 