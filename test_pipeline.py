"""
Pipeline Testing Script
Used to test the basic functionality of each module in the pipeline.
"""
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add the current directory to the Python path
sys.path.append(str(Path(__file__).parent))

def test_data_loader():
    """Tests the data loading module"""
    print("Testing Data Loader...")
    try:
        from data_loader import DataLoader
        
        loader = DataLoader()
        
        # Test loading diabetic data
        diabetic_data = loader.load_diabetic_data()
        assert isinstance(diabetic_data, pd.DataFrame)
        assert len(diabetic_data) > 0
        print("✅ Diabetic data loaded successfully")
        
        # Test loading ID mapping data
        ids_mapping = loader.load_ids_mapping()
        assert isinstance(ids_mapping, pd.DataFrame)
        assert len(ids_mapping) > 0
        print("✅ ID mapping data loaded successfully")
        
        # Test splitting ID mapping data
        admission_type, discharge_disposition, admission_source = loader.split_ids_mapping()
        assert len(admission_type) > 0
        assert len(discharge_disposition) > 0
        assert len(admission_source) > 0
        print("✅ ID mapping data split successfully")
        
        # Test merging data
        merged_data = loader.merge_data()
        assert isinstance(merged_data, pd.DataFrame)
        assert len(merged_data) > 0
        print("✅ Data merged successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        return False

def test_data_preprocessor():
    """Tests the data preprocessing module"""
    print("\nTesting Data Preprocessor...")
    try:
        from data_preprocessor import DataPreprocessor
        from data_loader import DataLoader
        
        # Load data
        loader = DataLoader()
        df = loader.merge_data()
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Test feature engineering
        df_engineered = preprocessor.apply_feature_engineering(df)
        assert len(df_engineered.columns) > len(df.columns)
        print("✅ Feature engineering completed")
        
        # Test target variable preparation
        df_with_target = preprocessor.prepare_target_variable(df_engineered)
        assert 'readmitted_binary' in df_with_target.columns
        print("✅ Target variable prepared")
        
        # Test data splitting
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
    """Tests the feature selection module"""
    print("\nTesting Feature Selector...")
    try:
        from feature_selector import FeatureSelector
        from data_preprocessor import DataPreprocessor
        from data_loader import DataLoader
        
        # Prepare data
        loader = DataLoader()
        df = loader.merge_data()
        
        preprocessor = DataPreprocessor()
        df = preprocessor.apply_feature_engineering(df)
        df = preprocessor.prepare_target_variable(df)
        
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(df)
        X_train, X_val, X_test = preprocessor.encode_categorical_features(X_train, X_val, X_test)
        X_train, X_val, X_test = preprocessor.scale_numerical_features(X_train, X_val, X_test)
        X_train_balanced, y_train_balanced = preprocessor.apply_smote(X_train, y_train)
        
        # Test feature selector
        selector = FeatureSelector()
        
        # Test L1 feature selection
        l1_features = selector.select_features_by_l1(X_train_balanced, y_train_balanced, top_n=5)
        assert len(l1_features) == 5
        print("✅ L1 feature selection completed")
        
        # Test mutual information feature selection
        mi_features = selector.select_features_by_mi(X_train_balanced, y_train_balanced, top_n=5)
        assert len(mi_features) == 5
        print("✅ Mutual Information feature selection completed")
        
        # Test tree-based feature selection
        tree_features = selector.select_features_by_tree(X_train_balanced, y_train_balanced, top_n=5)
        assert len(tree_features) == 5
        print("✅ Tree-based feature selection completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature selector test failed: {e}")
        return False

def test_model_trainer():
    """Tests the model training module"""
    print("\nTesting Model Trainer...")
    try:
        from model_trainer import ModelTrainer
        from feature_selector import FeatureSelector
        from data_preprocessor import DataPreprocessor
        from data_loader import DataLoader
        
        # Prepare data
        loader = DataLoader()
        df = loader.merge_data()
        
        preprocessor = DataPreprocessor()
        df = preprocessor.apply_feature_engineering(df)
        df = preprocessor.prepare_target_variable(df)
        
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(df)
        X_train, X_val, X_test = preprocessor.encode_categorical_features(X_train, X_val, X_test)
        X_train, X_val, X_test = preprocessor.scale_numerical_features(X_train, X_val, X_test)
        X_train_balanced, y_train_balanced = preprocessor.apply_smote(X_train, y_train)
        
        # Feature selection
        selector = FeatureSelector()
        selected_features = selector.select_all_features(X_train_balanced, y_train_balanced, top_n=10)
        best_features = selected_features['MutualInfo']
        
        X_train_selected = X_train_balanced[best_features]
        X_val_selected = X_val[best_features]
        X_test_selected = X_test[best_features]
        
        # Test model trainer
        trainer = ModelTrainer()
        
        # Test getting models
        models = trainer.get_models()
        assert len(models) > 0
        print("✅ Models initialized")
        
        # Test single model training
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
    """Tests the main pipeline"""
    print("\nTesting Main Pipeline...")
    try:
        from main_pipeline import HospitalReadmissionPipeline
        
        # Initialize pipeline
        pipeline = HospitalReadmissionPipeline()
        assert pipeline is not None
        print("✅ Pipeline initialized")
        
        # Test data loading step
        raw_data = pipeline.run_data_loading()
        assert isinstance(raw_data, pd.DataFrame)
        assert len(raw_data) > 0
        print("✅ Pipeline data loading completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Main pipeline test failed: {e}")
        return False

def main():
    """Runs all tests"""
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