"""
Validation使用示例 - 展示在demo中的具体调用位置
"""
import pandas as pd
import numpy as np
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from feature_selector import FeatureSelector
from model_trainer import ModelTrainer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validation_usage_demo():
    """
    展示validation在demo中的具体使用
    """
    logger.info("=" * 60)
    logger.info("VALIDATION使用示例")
    logger.info("=" * 60)
    
    # ==================== 第一步：数据分割 ====================
    logger.info("\n📊 第一步：数据分割 (Train/Validation/Test)")
    
    # 加载和预处理数据
    loader = DataLoader()
    df = loader.merge_data()
    
    preprocessor = DataPreprocessor()
    df = preprocessor.apply_feature_engineering(df)
    df = preprocessor.prepare_target_variable(df)
    
    # 🔑 关键：三路数据分割
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        df, target_col='readmitted_binary', 
        test_size=0.2,  # 20% 测试集
        val_size=0.2,   # 20% 验证集 (从剩余80%中取20%，即总数据的16%)
        random_state=42
    )
    
    logger.info(f"数据分割结果:")
    logger.info(f"  训练集: {X_train.shape} ({len(X_train)/len(df)*100:.1f}%)")
    logger.info(f"  验证集: {X_val.shape} ({len(X_val)/len(df)*100:.1f}%)")
    logger.info(f"  测试集: {X_test.shape} ({len(X_test)/len(df)*100:.1f}%)")
    
    # 继续预处理
    X_train, X_val, X_test = preprocessor.encode_categorical_features(X_train, X_val, X_test)
    X_train, X_val, X_test = preprocessor.scale_numerical_features(X_train, X_val, X_test)
    X_train_balanced, y_train_balanced = preprocessor.apply_smote(X_train, y_train)
    
    # ==================== 第二步：特征选择 ====================
    logger.info("\n🔍 第二步：特征选择 (使用训练集)")
    
    feature_selector = FeatureSelector()
    
    # 只在训练集上进行特征选择
    selected_features = feature_selector.select_features_mutual_info(
        X_train_balanced, y_train_balanced, top_n=20
    )
    
    logger.info(f"选择了 {len(selected_features)} 个特征")
    
    # 应用特征选择到所有数据集
    X_train_selected = X_train_balanced[selected_features]
    X_val_selected = X_val[selected_features]      # 🔑 验证集使用相同特征
    X_test_selected = X_test[selected_features]    # 🔑 测试集使用相同特征
    
    # ==================== 第三步：模型训练 ====================
    logger.info("\n🤖 第三步：模型训练 (在训练集上)")
    
    model_trainer = ModelTrainer(random_state=42)
    
    # 在训练集上训练模型
    training_results = model_trainer.train_all_models(
        X_train_selected, y_train_balanced, 
        X_val_selected, y_val  # 🔑 传入验证集用于概率分布图
    )
    
    logger.info("训练结果 (Cross-Validation):")
    logger.info(training_results.to_string())
    
    # ==================== 第四步：验证集评估 ====================
    logger.info("\n✅ 第四步：验证集评估 (模型选择)")
    
    # 🔑 关键：在验证集上评估所有模型
    validation_results = model_trainer.evaluate_on_validation_set(X_val_selected, y_val)
    
    logger.info("验证集结果:")
    logger.info(validation_results.to_string())
    
    # ==================== 第五步：模型选择 ====================
    logger.info("\n🏆 第五步：模型选择 (基于验证集性能)")
    
    # 基于验证集AUC选择最佳模型
    best_val_model_name = validation_results.loc[validation_results['auc'].idxmax(), 'model_name']
    best_val_auc = validation_results.loc[validation_results['auc'].idxmax(), 'auc']
    
    logger.info(f"验证集最佳模型: {best_val_model_name} (AUC: {best_val_auc:.3f})")
    
    # 基于验证集F1选择最佳模型
    best_val_f1_model_name = validation_results.loc[validation_results['f1'].idxmax(), 'model_name']
    best_val_f1 = validation_results.loc[validation_results['f1'].idxmax(), 'f1']
    
    logger.info(f"验证集最佳模型 (F1): {best_val_f1_model_name} (F1: {best_val_f1:.3f})")
    
    # ==================== 第六步：测试集评估 ====================
    logger.info("\n🎯 第六步：测试集评估 (最终性能)")
    
    # 🔑 关键：只在测试集上评估最终选择的模型
    test_results = model_trainer.evaluate_on_test_set(X_test_selected, y_test)
    
    logger.info("测试集结果:")
    logger.info(test_results.to_string())
    
    # ==================== 第七步：过拟合检测 ====================
    logger.info("\n🔍 第七步：过拟合检测")
    
    # 比较验证集和测试集性能
    for _, row in validation_results.iterrows():
        model_name = row['model_name']
        val_auc = row['auc']
        val_f1 = row['f1']
        
        # 找到对应的测试集结果
        test_row = test_results[test_results['model_name'] == model_name]
        if not test_row.empty:
            test_auc = test_row.iloc[0]['auc']
            test_f1 = test_row.iloc[0]['f1']
            
            val_test_auc_diff = abs(val_auc - test_auc)
            val_test_f1_diff = abs(val_f1 - test_f1)
            
            logger.info(f"\n{model_name}:")
            logger.info(f"  验证集: AUC={val_auc:.3f}, F1={val_f1:.3f}")
            logger.info(f"  测试集: AUC={test_auc:.3f}, F1={test_f1:.3f}")
            logger.info(f"  差异: AUC差异={val_test_auc_diff:.3f}, F1差异={val_test_f1_diff:.3f}")
            
            if val_test_auc_diff > 0.05:
                logger.warning(f"  ⚠️  {model_name} 可能存在过拟合!")
            else:
                logger.info(f"  ✅ {model_name} 泛化性能良好")
    
    # ==================== 第八步：超参数调优示例 ====================
    logger.info("\n⚙️ 第八步：超参数调优示例")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    
    # 为RandomForest调优n_estimators参数
    n_estimators_list = [50, 100, 200, 300]
    val_scores = []
    
    logger.info("RandomForest超参数调优:")
    for n_estimators in n_estimators_list:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train_selected, y_train_balanced)
        
        # 🔑 关键：在验证集上评估
        y_val_pred_proba = model.predict_proba(X_val_selected)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_pred_proba)
        val_scores.append(val_auc)
        
        logger.info(f"  n_estimators={n_estimators}: 验证集AUC={val_auc:.3f}")
    
    best_n_estimators = n_estimators_list[np.argmax(val_scores)]
    logger.info(f"最佳n_estimators: {best_n_estimators}")
    
    # ==================== 总结 ====================
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION使用总结")
    logger.info("=" * 60)
    
    logger.info("""
    📋 Validation的正确使用流程:
    
    1. 数据分割: Train(64%) + Validation(16%) + Test(20%)
    2. 特征选择: 只在训练集上进行
    3. 模型训练: 在训练集上训练
    4. 验证集评估: 评估所有模型，选择最佳模型
    5. 超参数调优: 在验证集上评估不同参数
    6. 测试集评估: 只在最终选择的模型上进行
    7. 过拟合检测: 比较验证集和测试集性能
    
    ⚠️ 重要原则:
    - 验证集只用于模型选择和超参数调优
    - 测试集只用于最终评估，不能用于任何调优
    - 特征选择必须在训练集上进行
    """)
    
    return {
        'training_results': training_results,
        'validation_results': validation_results,
        'test_results': test_results,
        'best_model': best_val_model_name
    }

if __name__ == "__main__":
    results = validation_usage_demo() 