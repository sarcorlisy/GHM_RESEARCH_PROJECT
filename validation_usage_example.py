"""
Validation Usage Example - Demonstrates the specific usage location in the demo
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
    Demonstrate the specific usage of validation in the demo
    """
    logger.info("=" * 60)
    logger.info("VALIDATION USAGE EXAMPLE")
    logger.info("=" * 60)
    
    # ==================== Step 1: Data Splitting ====================
    logger.info("\n📊 Step 1: Data Splitting (Train/Validation/Test)")
    
    # Load and preprocess data
    loader = DataLoader()
    df = loader.merge_data()
    
    preprocessor = DataPreprocessor()
    df = preprocessor.apply_feature_engineering(df)
    df = preprocessor.prepare_target_variable(df)
    
    # 🔑 Key: Three-way data split
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        df, target_col='readmitted_binary', 
        test_size=0.2,  # 20% test set
        val_size=0.2,   # 20% validation set (from remaining 80%, i.e., 16% of total)
        random_state=42
    )
    
    logger.info(f"Data split results:")
    logger.info(f"  Train set: {X_train.shape} ({len(X_train)/len(df)*100:.1f}%)")
    logger.info(f"  Validation set: {X_val.shape} ({len(X_val)/len(df)*100:.1f}%)")
    logger.info(f"  Test set: {X_test.shape} ({len(X_test)/len(df)*100:.1f}%)")
    
    # Continue preprocessing
    X_train, X_val, X_test = preprocessor.encode_categorical_features(X_train, X_val, X_test)
    X_train, X_val, X_test = preprocessor.scale_numerical_features(X_train, X_val, X_test)
    X_train_balanced, y_train_balanced = preprocessor.apply_smote(X_train, y_train)
    
    # ==================== Step 2: Feature Selection ====================
    logger.info("\n🔍 Step 2: Feature Selection (using training set)")
    
    feature_selector = FeatureSelector()
    
    # Feature selection only on training set
    selected_features = feature_selector.select_features_mutual_info(
        X_train_balanced, y_train_balanced, top_n=20
    )
    
    logger.info(f"Selected {len(selected_features)} features")
    
    # Apply feature selection to all datasets
    X_train_selected = X_train_balanced[selected_features]
    X_val_selected = X_val[selected_features]      # 🔑 Validation set uses same features
    X_test_selected = X_test[selected_features]    # 🔑 Test set uses same features
    
    # ==================== Step 3: Model Training ====================
    logger.info("\n🤖 Step 3: Model Training (on training set)")
    
    model_trainer = ModelTrainer(random_state=42)
    
    # Train models on training set
    training_results = model_trainer.train_all_models(
        X_train_selected, y_train_balanced, 
        X_val_selected, y_val  # 🔑 Pass validation set for probability plot
    )
    
    logger.info("Training results (Cross-Validation):")
    logger.info(training_results.to_string())
    
    # ==================== Step 4: Validation Set Evaluation ====================
    logger.info("\n✅ Step 4: Validation Set Evaluation (Model Selection)")
    
    # 🔑 Key: Evaluate all models on validation set
    validation_results = model_trainer.evaluate_on_validation_set(X_val_selected, y_val)
    
    logger.info("Validation set results:")
    logger.info(validation_results.to_string())
    
    # ==================== Step 5: Model Selection ====================
    logger.info("\n🏆 Step 5: Model Selection (Based on Validation Set Performance)")
    
    # Select best model by validation set AUC
    best_val_model_name = validation_results.loc[validation_results['auc'].idxmax(), 'model_name']
    best_val_auc = validation_results.loc[validation_results['auc'].idxmax(), 'auc']
    
    logger.info(f"Best model on validation set: {best_val_model_name} (AUC: {best_val_auc:.3f})")
    
    # Select best model by validation set F1
    best_val_f1_model_name = validation_results.loc[validation_results['f1'].idxmax(), 'model_name']
    best_val_f1 = validation_results.loc[validation_results['f1'].idxmax(), 'f1']
    
    logger.info(f"Best model on validation set (F1): {best_val_f1_model_name} (F1: {best_val_f1:.3f})")
    
    # ==================== Step 6: Test Set Evaluation ====================
    logger.info("\n🎯 Step 6: Test Set Evaluation (Final Performance)")
    
    # 🔑 Key: Only evaluate the final selected model on the test set
    test_results = model_trainer.evaluate_on_test_set(X_test_selected, y_test)
    
    logger.info("Test set results:")
    logger.info(test_results.to_string())
    
    # ==================== Step 7: Overfitting Detection ====================
    logger.info("\n🔍 Step 7: Overfitting Detection")
    
    # Compare validation and test set performance
    for _, row in validation_results.iterrows():
        model_name = row['model_name']
        val_auc = row['auc']
        val_f1 = row['f1']
        
        # Find corresponding test set result
        test_row = test_results[test_results['model_name'] == model_name]
        if not test_row.empty:
            test_auc = test_row.iloc[0]['auc']
            test_f1 = test_row.iloc[0]['f1']
            
            val_test_auc_diff = abs(val_auc - test_auc)
            val_test_f1_diff = abs(val_f1 - test_f1)
            
            logger.info(f"\n{model_name}:")
            logger.info(f"  Validation set: AUC={val_auc:.3f}, F1={val_f1:.3f}")
            logger.info(f"  Test set: AUC={test_auc:.3f}, F1={test_f1:.3f}")
            logger.info(f"  Difference: AUC diff={val_test_auc_diff:.3f}, F1 diff={val_test_f1_diff:.3f}")
            
            if val_test_auc_diff > 0.05:
                logger.warning(f"  ⚠️  {model_name} may be overfitting!")
            else:
                logger.info(f"  ✅ {model_name} has good generalization performance")
    
    # ==================== Step 8: Hyperparameter Tuning Example ====================
    logger.info("\n⚙️ Step 8: Hyperparameter Tuning Example")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    
    # Tune n_estimators parameter for RandomForest
    n_estimators_list = [50, 100, 200, 300]
    val_scores = []
    
    logger.info("RandomForest hyperparameter tuning:")
    for n_estimators in n_estimators_list:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train_selected, y_train_balanced)
        
        # 🔑 Key: Evaluate on validation set
        y_val_pred_proba = model.predict_proba(X_val_selected)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_pred_proba)
        val_scores.append(val_auc)
        
        logger.info(f"  n_estimators={n_estimators}: Validation set AUC={val_auc:.3f}")
    
    best_n_estimators = n_estimators_list[np.argmax(val_scores)]
    logger.info(f"Best n_estimators: {best_n_estimators}")
    
    # ==================== Summary ====================
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION USAGE SUMMARY")
    logger.info("=" * 60)
    
    logger.info("""
    📋 Correct usage process of validation:
    
    1. Data splitting: Train(64%) + Validation(16%) + Test(20%)
    2. Feature selection: Only on training set
    3. Model training: Train on training set
    4. Validation set evaluation: Evaluate all models, select the best model
    5. Hyperparameter tuning: Evaluate different parameters on validation set
    6. Test set evaluation: Only on the final selected model
    7. Overfitting detection: Compare validation and test set performance
    
    ⚠️ Important principles:
    - Validation set is only for model selection and hyperparameter tuning
    - Test set is only for final evaluation, cannot be used for any tuning
    - Feature selection must be performed on the training set
    """)
    
    return {
        'training_results': training_results,
        'validation_results': validation_results,
        'test_results': test_results,
        'best_model': best_val_model_name
    }

if __name__ == "__main__":
    results = validation_usage_demo() 