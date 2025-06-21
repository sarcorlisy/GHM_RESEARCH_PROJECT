"""
Pipeline Testing Script
用于测试pipeline各个模块的基本功能
"""
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

def test_data_loader():
    """测试数据加载模块"""
    print("Testing Data Loader...")
    try:
        from data_loader import DataLoader
        
        loader = DataLoader()
        
        # 测试加载糖尿病数据
        diabetic_data = loader.load_diabetic_data()
        assert isinstance(diabetic_data, pd.DataFrame)
        assert len(diabetic_data) > 0
        print("✅ Diabetic data loaded successfully")
        
        # 测试加载ID映射数据
        ids_mapping = loader.load_ids_mapping()
        assert isinstance(ids_mapping, pd.DataFrame)
        assert len(ids_mapping) > 0
        print("✅ ID mapping data loaded successfully")
        
        # 测试分割ID映射数据
        admission_type, discharge_disposition, admission_source = loader.split_ids_mapping()
        assert len(admission_type) > 0
        assert len(discharge_disposition) > 0
        assert len(admission_source) > 0
        print("✅ ID mapping data split successfully")
        
        # 测试合并数据
        merged_data = loader.merge_data()
        assert isinstance(merged_data, pd.DataFrame)
        assert len(merged_data) > 0
        print("✅ Data merged successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        return False

def test_data_preprocessor():
    """测试数据预处理模块"""
    print("\nTesting Data Preprocessor...")
    try:
        from data_preprocessor import DataPreprocessor
        from data_loader import DataLoader
        
        # 加载数据
        loader = DataLoader()
        df = loader.merge_data()
        
        # 初始化预处理器
        preprocessor = DataPreprocessor()
        
        # 测试特征工程
        df_engineered = preprocessor.apply_feature_engineering(df)
        assert len(df_engineered.columns) > len(df.columns)
        print("✅ Feature engineering completed")
        
        # 测试目标变量准备
        df_with_target = preprocessor.prepare_target_variable(df_engineered)
        assert 'readmitted_binary' in df_with_target.columns
        print("✅ Target variable prepared")
        
        # 测试数据分割
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(df_with_target)
        assert len(X_train) > 0
        assert len(X_val) > 0
        assert len(X_test) > 0
        print("✅ Data split completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Data preprocessor test failed: {e}")
        return False

def test_feature_selector():
    """测试特征选择模块"""
    print("\nTesting Feature Selector...")
    try:
        from feature_selector import FeatureSelector
        from data_preprocessor import DataPreprocessor
        from data_loader import DataLoader
        
        # 准备数据
        loader = DataLoader()
        df = loader.merge_data()
        
        preprocessor = DataPreprocessor()
        df = preprocessor.apply_feature_engineering(df)
        df = preprocessor.prepare_target_variable(df)
        
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(df)
        X_train, X_val, X_test = preprocessor.encode_categorical_features(X_train, X_val, X_test)
        X_train, X_val, X_test = preprocessor.scale_numerical_features(X_train, X_val, X_test)
        X_train_balanced, y_train_balanced = preprocessor.apply_smote(X_train, y_train)
        
        # 测试特征选择器
        selector = FeatureSelector()
        
        # 测试L1特征选择
        l1_features = selector.select_features_by_l1(X_train_balanced, y_train_balanced, top_n=5)
        assert len(l1_features) == 5
        print("✅ L1 feature selection completed")
        
        # 测试互信息特征选择
        mi_features = selector.select_features_by_mi(X_train_balanced, y_train_balanced, top_n=5)
        assert len(mi_features) == 5
        print("✅ Mutual Information feature selection completed")
        
        # 测试树模型特征选择
        tree_features = selector.select_features_by_tree(X_train_balanced, y_train_balanced, top_n=5)
        assert len(tree_features) == 5
        print("✅ Tree-based feature selection completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature selector test failed: {e}")
        return False

def test_model_trainer():
    """测试模型训练模块"""
    print("\nTesting Model Trainer...")
    try:
        from model_trainer import ModelTrainer
        from feature_selector import FeatureSelector
        from data_preprocessor import DataPreprocessor
        from data_loader import DataLoader
        
        # 准备数据
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
        selector = FeatureSelector()
        selected_features = selector.select_all_features(X_train_balanced, y_train_balanced, top_n=10)
        best_features = selected_features['MutualInfo']
        
        X_train_selected = X_train_balanced[best_features]
        X_val_selected = X_val[best_features]
        X_test_selected = X_test[best_features]
        
        # 测试模型训练器
        trainer = ModelTrainer()
        
        # 测试获取模型
        models = trainer.get_models()
        assert len(models) > 0
        print("✅ Models initialized")
        
        # 测试单个模型训练
        result = trainer.train_single_model('RandomForest', X_train_selected, y_train_balanced, 
                                          X_val_selected, y_val)
        assert 'cv_auc' in result
        assert 'cv_f1' in result
        print("✅ Single model training completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Model trainer test failed: {e}")
        return False

def test_main_pipeline():
    """测试主pipeline"""
    print("\nTesting Main Pipeline...")
    try:
        from main_pipeline import HospitalReadmissionPipeline
        
        # 初始化pipeline
        pipeline = HospitalReadmissionPipeline()
        assert pipeline is not None
        print("✅ Pipeline initialized")
        
        # 测试数据加载步骤
        raw_data = pipeline.run_data_loading()
        assert isinstance(raw_data, pd.DataFrame)
        assert len(raw_data) > 0
        print("✅ Pipeline data loading completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Main pipeline test failed: {e}")
        return False

def main():
    """运行所有测试"""
    print("🧪 Pipeline Testing Suite")
    print("=" * 40)
    
    tests = [
        ("Data Loader", test_data_loader),
        ("Data Preprocessor", test_data_preprocessor),
        ("Feature Selector", test_feature_selector),
        ("Model Trainer", test_model_trainer),
        ("Main Pipeline", test_main_pipeline)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} test passed")
        else:
            print(f"❌ {test_name} test failed")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Pipeline is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main() 