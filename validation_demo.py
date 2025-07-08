"""
Validation Demo - 展示完整的验证流程
"""
import pandas as pd
import numpy as np
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from feature_selector import FeatureSelector
from model_trainer import ModelTrainer
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_selected_features_dict(feature_selectors, X_train, y_train, top_n_list):
    """
    生成所有特征选择方法和top_n组合的特征子集字典。
    Args:
        feature_selectors: dict, 特征选择方法名到函数的映射
        X_train, y_train: 训练集
        top_n_list: list, top_n的取值
    Returns:
        dict: (fs_name, top_n) -> feature_list
    """
    selected_features_dict = {}
    for fs_name, fs_func in feature_selectors.items():
        for top_n in top_n_list:
            features = fs_func(X_train, y_train, top_n=top_n)
            selected_features_dict[(fs_name, top_n)] = features
    return selected_features_dict

def run_validation_demo():
    """
    运行完整的validation demo
    """
    logger.info("=" * 60)
    logger.info("VALIDATION DEMO - 医院再入院预测验证流程")
    logger.info("=" * 60)
    
    # 1. 数据加载
    logger.info("\n1. 数据加载...")
    loader = DataLoader()
    df = loader.merge_data()
    logger.info(f"原始数据形状: {df.shape}")
    
    # 2. 数据预处理
    logger.info("\n2. 数据预处理...")
    preprocessor = DataPreprocessor()
    
    # 特征工程
    df = preprocessor.apply_feature_engineering(df)
    logger.info(f"特征工程后数据形状: {df.shape}")
    
    # 准备目标变量
    df = preprocessor.prepare_target_variable(df)
    logger.info(f"目标变量分布:\n{df['readmitted_binary'].value_counts()}")
    
    # 数据分割 (Train/Validation/Test)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        df, target_col='readmitted_binary', test_size=0.2, val_size=0.2, random_state=42
    )
    
    logger.info(f"数据分割结果:")
    logger.info(f"  训练集: {X_train.shape}")
    logger.info(f"  验证集: {X_val.shape}")
    logger.info(f"  测试集: {X_test.shape}")
    
    # 特征编码
    X_train, X_val, X_test = preprocessor.encode_categorical_features(
        X_train, X_val, X_test, encoding_method='label'
    )
    
    # 特征标准化
    X_train, X_val, X_test = preprocessor.scale_numerical_features(X_train, X_val, X_test)
    
    # SMOTE平衡
    X_train_balanced, y_train_balanced = preprocessor.apply_smote(X_train, y_train)
    logger.info(f"SMOTE平衡后训练集形状: {X_train_balanced.shape}")
    
    # 3. 特征选择
    logger.info("\n3. 特征选择...")
    feature_selector = FeatureSelector()
    feature_selectors = feature_selector.get_feature_selectors()
    top_n_list = [5, 10, 15]

    # 新增：保存所有方法和top_n的特征子集
    selected_features_dict = get_selected_features_dict(feature_selectors, X_train_balanced, y_train_balanced, top_n_list)
    logger.info(f"已保存所有特征选择方法和top_n组合的特征子集，共{len(selected_features_dict)}组")
    
    # 4. 模型训练和验证
    logger.info("\n4. 模型训练和验证...")
    model_trainer = ModelTrainer(random_state=42)
    
    # 使用最佳特征集进行训练
    best_features = selected_features_dict[('MutualInfo', 10)]
    logger.info(f"使用 MutualInfo 选择的 {len(best_features)} 个特征")
    
    X_train_selected = X_train_balanced[best_features]
    X_val_selected = X_val[best_features]
    X_test_selected = X_test[best_features]
    
    # 训练所有模型
    training_results = model_trainer.train_all_models(
        X_train_selected, y_train_balanced, X_val_selected, y_val
    )
    
    logger.info("\n训练结果 (Cross-Validation):")
    logger.info(training_results.to_string())
    
    # 5. 验证集评估
    logger.info("\n5. 验证集评估...")
    validation_results = model_trainer.evaluate_on_validation_set(X_val_selected, y_val)
    
    logger.info("验证集结果:")
    logger.info(validation_results.to_string())
    
    # 6. 测试集评估
    logger.info("\n6. 测试集评估...")
    test_results = model_trainer.evaluate_on_test_set(X_test_selected, y_test)
    
    logger.info("测试集结果:")
    logger.info(test_results.to_string())
    
    # 7. 结果比较
    logger.info("\n7. 结果比较...")
    logger.info("=" * 80)
    logger.info("模型性能比较 (CV vs Validation vs Test)")
    logger.info("=" * 80)
    
    # 合并结果进行比较
    comparison_df = pd.merge(
        training_results[['model_name', 'cv_auc', 'cv_f1']], 
        validation_results[['model_name', 'auc', 'f1']], 
        on='model_name', suffixes=('_cv', '_val')
    )
    comparison_df = pd.merge(
        comparison_df,
        test_results[['model_name', 'auc', 'f1']],
        on='model_name', suffixes=('', '_test')
    )
    comparison_df = comparison_df.rename(columns={'auc': 'auc_test', 'f1': 'f1_test'})
    
    logger.info(comparison_df.to_string(index=False))
    
    # 8. 找出最佳模型
    logger.info("\n8. 最佳模型分析...")
    
    # 按验证集AUC选择最佳模型
    best_val_model = validation_results.loc[validation_results['auc'].idxmax()]
    logger.info(f"验证集最佳模型 (AUC): {best_val_model['model_name']} (AUC: {best_val_model['auc']:.3f})")
    
    # 按验证集F1选择最佳模型
    best_val_f1_model = validation_results.loc[validation_results['f1'].idxmax()]
    logger.info(f"验证集最佳模型 (F1): {best_val_f1_model['model_name']} (F1: {best_val_f1_model['f1']:.3f})")
    
    # 检查过拟合
    logger.info("\n9. 过拟合检查...")
    for _, row in comparison_df.iterrows():
        model_name = row['model_name']
        cv_auc = row['cv_auc']
        val_auc = row['auc_val']
        test_auc = row['auc_test']
        
        # 检查验证集和测试集的性能差异
        val_test_diff = abs(val_auc - test_auc)
        cv_val_diff = abs(cv_auc - val_auc)
        
        logger.info(f"{model_name}:")
        logger.info(f"  CV AUC: {cv_auc:.3f}, Val AUC: {val_auc:.3f}, Test AUC: {test_auc:.3f}")
        logger.info(f"  CV-Val差异: {cv_val_diff:.3f}, Val-Test差异: {val_test_diff:.3f}")
        
        if val_test_diff > 0.05:
            logger.warning(f"  ⚠️  {model_name} 可能存在过拟合 (Val-Test差异: {val_test_diff:.3f})")
        else:
            logger.info(f"  ✅ {model_name} 泛化性能良好")
    
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION DEMO 完成!")
    logger.info("=" * 60)
    
    return {
        'training_results': training_results,
        'validation_results': validation_results,
        'test_results': test_results,
        'comparison_df': comparison_df
    }

if __name__ == "__main__":
    run_validation_demo() 