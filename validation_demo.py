"""
Validation Demo - Demonstrates the complete validation process
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
    Generate a dictionary of feature subsets for all feature selection methods and top_n combinations.
    Args:
        feature_selectors: dict, mapping from feature selection method name to function
        X_train, y_train: Training set
        top_n_list: list, values of top_n
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
    Run the complete validation demo
    """
    logger.info("=" * 60)
    logger.info("VALIDATION DEMO - Hospital Readmission Prediction Validation Process")
    logger.info("=" * 60)
    
    # 1. Data loading
    logger.info("\n1. Data loading...")
    loader = DataLoader()
    df = loader.merge_data()
    logger.info(f"Original data shape: {df.shape}")
    
    # 2. Data preprocessing
    logger.info("\n2. Data preprocessing...")
    preprocessor = DataPreprocessor()
    
    # Feature engineering
    df = preprocessor.apply_feature_engineering(df)
    logger.info(f"Data shape after feature engineering: {df.shape}")
    
    # Prepare target variable
    df = preprocessor.prepare_target_variable(df)
    logger.info(f"Target variable distribution:\n{df['readmitted_binary'].value_counts()}")
    
    # Data splitting (Train/Validation/Test)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        df, target_col='readmitted_binary', test_size=0.2, val_size=0.2, random_state=42
    )
    
    logger.info(f"Data split results:")
    logger.info(f"  Train set: {X_train.shape}")
    logger.info(f"  Validation set: {X_val.shape}")
    logger.info(f"  Test set: {X_test.shape}")
    
    # Feature encoding
    X_train, X_val, X_test = preprocessor.encode_categorical_features(
        X_train, X_val, X_test, encoding_method='label'
    )
    
    # Feature standardization
    X_train, X_val, X_test = preprocessor.scale_numerical_features(X_train, X_val, X_test)
    
    # SMOTE balancing
    X_train_balanced, y_train_balanced = preprocessor.apply_smote(X_train, y_train)
    logger.info(f"Training set shape after SMOTE balancing: {X_train_balanced.shape}")
    
    # 3. Feature selection
    logger.info("\n3. Feature selection...")
    feature_selector = FeatureSelector()
    feature_selectors = feature_selector.get_feature_selectors()
    top_n_list = [5, 10, 15]

    # New: Save feature subsets for all methods and top_n
    selected_features_dict = get_selected_features_dict(feature_selectors, X_train_balanced, y_train_balanced, top_n_list)
    logger.info(f"Saved all feature subsets for feature selection methods and top_n combinations, total {len(selected_features_dict)} groups")
    
    # 4. Model training and validation
    logger.info("\n4. Model training and validation...")
    model_trainer = ModelTrainer(random_state=42)
    
    # Train using the best feature set
    best_features = selected_features_dict[('MutualInfo', 10)]
    logger.info(f"Using {len(best_features)} features selected by MutualInfo")
    
    X_train_selected = X_train_balanced[best_features]
    X_val_selected = X_val[best_features]
    X_test_selected = X_test[best_features]
    
    # Train all models
    training_results = model_trainer.train_all_models(
        X_train_selected, y_train_balanced, X_val_selected, y_val
    )
    
    logger.info("\nTraining results (Cross-Validation):")
    logger.info(training_results.to_string())
    
    # 5. Validation set evaluation
    logger.info("\n5. Validation set evaluation...")
    validation_results = model_trainer.evaluate_on_validation_set(X_val_selected, y_val)
    
    logger.info("Validation set results:")
    logger.info(validation_results.to_string())
    
    # 6. Test set evaluation
    logger.info("\n6. Test set evaluation...")
    test_results = model_trainer.evaluate_on_test_set(X_test_selected, y_test)
    
    logger.info("Test set results:")
    logger.info(test_results.to_string())
    
    # 7. Results comparison
    logger.info("\n7. Results comparison...")
    logger.info("=" * 80)
    logger.info("Model performance comparison (CV vs Validation vs Test)")
    logger.info("=" * 80)
    
    # Merge results for comparison
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
    
    # 8. Find the best model
    logger.info("\n8. Best model analysis...")
    
    # Select the best model by validation set AUC
    best_val_model = validation_results.loc[validation_results['auc'].idxmax()]
    logger.info(f"Best model on validation set (AUC): {best_val_model['model_name']} (AUC: {best_val_model['auc']:.3f})")
    
    # Select the best model by validation set F1
    best_val_f1_model = validation_results.loc[validation_results['f1'].idxmax()]
    logger.info(f"Best model on validation set (F1): {best_val_f1_model['model_name']} (F1: {best_val_f1_model['f1']:.3f})")
    
    # Overfitting check
    logger.info("\n9. Overfitting check...")
    for _, row in comparison_df.iterrows():
        model_name = row['model_name']
        cv_auc = row['cv_auc']
        val_auc = row['auc_val']
        test_auc = row['auc_test']
        
        # Check performance difference between validation and test sets
        val_test_diff = abs(val_auc - test_auc)
        cv_val_diff = abs(cv_auc - val_auc)
        
        logger.info(f"{model_name}:")
        logger.info(f"  CV AUC: {cv_auc:.3f}, Val AUC: {val_auc:.3f}, Test AUC: {test_auc:.3f}")
        logger.info(f"  CV-Val difference: {cv_val_diff:.3f}, Val-Test difference: {val_test_diff:.3f}")
        
        if val_test_diff > 0.05:
            logger.warning(f"  ⚠️  {model_name} may be overfitting (Val-Test difference: {val_test_diff:.3f})")
        else:
            logger.info(f"  ✅ {model_name} has good generalization performance")
    
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION DEMO COMPLETED!")
    logger.info("=" * 60)
    
    return {
        'training_results': training_results,
        'validation_results': validation_results,
        'test_results': test_results,
        'comparison_df': comparison_df
    }

if __name__ == "__main__":
    run_validation_demo() 