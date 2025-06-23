"""
Main Pipeline for Hospital Readmission Prediction
End-to-end data science pipeline integrating all modules.
"""
import pandas as pd
import numpy as np
import logging
import argparse
from pathlib import Path
import warnings
from typing import Dict, Any, Optional

# Import custom modules
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from feature_selector import FeatureSelector
from model_trainer import ModelTrainer
from pipeline_config import MODEL_CONFIG, DATA_PATHS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

class HospitalReadmissionPipeline:
    """Main pipeline class for hospital readmission prediction"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the pipeline
        
        Args:
            config: A configuration dictionary
        """
        self.config = config or MODEL_CONFIG
        self.data_loader = None
        self.preprocessor = None
        self.feature_selector = None
        self.model_trainer = None
        
        # Store intermediate results
        self.raw_data = None
        self.processed_data = None
        self.selected_features = None
        self.training_results = None
        self.test_results = None
        
        logger.info("Hospital Readmission Pipeline initialized")
    
    def run_data_loading(self) -> pd.DataFrame:
        """
        Runs the data loading step
        
        Returns:
            The merged raw data
        """
        logger.info("=" * 50)
        logger.info("STEP 1: Data Loading")
        logger.info("=" * 50)
        
        self.data_loader = DataLoader()
        self.raw_data = self.data_loader.merge_data()
        
        # Get data information
        data_info = self.data_loader.get_data_info()
        logger.info(f"Data loaded successfully: {data_info['shape']}")
        logger.info(f"Number of features: {len(data_info['columns'])}")
        
        # Save the merged data
        self.data_loader.save_merged_data()
        
        return self.raw_data
    
    def run_data_preprocessing(self) -> tuple:
        """
        Runs the data preprocessing step
        
        Returns:
            Preprocessed train, validation, and test data
        """
        logger.info("=" * 50)
        logger.info("STEP 2: Data Preprocessing")
        logger.info("=" * 50)
        
        self.preprocessor = DataPreprocessor()
        
        # Apply feature engineering
        self.processed_data = self.preprocessor.apply_feature_engineering(self.raw_data)
        logger.info(f"Feature engineering completed. New shape: {self.processed_data.shape}")
        
        # Prepare the target variable
        self.processed_data = self.preprocessor.prepare_target_variable(self.processed_data)
        
        # Split the data
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.split_data(
            self.processed_data,
            test_size=self.config['test_size'],
            val_size=self.config['val_size'],
            random_state=self.config['random_state']
        )
        
        # Encode categorical features
        X_train, X_val, X_test = self.preprocessor.encode_categorical_features(X_train, X_val, X_test)
        
        # Scale numerical features
        X_train, X_val, X_test = self.preprocessor.scale_numerical_features(X_train, X_val, X_test)
        
        # Apply SMOTE to balance the dataset
        X_train_balanced, y_train_balanced = self.preprocessor.apply_smote(X_train, y_train)
        
        # Save the preprocessed data
        self.preprocessor.save_preprocessed_data(
            X_train_balanced, X_val, X_test, 
            y_train_balanced, y_val, y_test
        )
        
        logger.info("Data preprocessing completed successfully")
        return X_train_balanced, X_val, X_test, y_train_balanced, y_val, y_test
    
    def run_feature_selection(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, list]:
        """
        Runs the feature selection step
        
        Args:
            X_train: Training set features
            y_train: Training set target variable
            
        Returns:
            A dictionary of selected features
        """
        logger.info("=" * 50)
        logger.info("STEP 3: Feature Selection")
        logger.info("=" * 50)
        
        self.feature_selector = FeatureSelector()
        
        # Select features using all methods
        self.selected_features = self.feature_selector.select_all_features(
            X_train, y_train, 
            top_n=self.config['feature_selection_top_n']
        )
        
        # Print selection results
        for method, features in self.selected_features.items():
            logger.info(f"{method} selected {len(features)} features")
        
        # Get common features
        common_features = self.feature_selector.get_common_features(min_methods=2)
        logger.info(f"Common features selected by at least 2 methods: {len(common_features)}")
        
        # Save selected features
        self.feature_selector.save_selected_features()
        
        # Plot feature importance
        self.feature_selector.plot_feature_importance(save_path="outputs/feature_importance.png")
        
        return self.selected_features
    
    def run_model_training(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series,
                          selected_features: Dict[str, list]) -> tuple:
        """
        Runs the model training step
        
        Args:
            X_train: Training set features
            y_train: Training set target variable
            X_val: Validation set features
            y_val: Validation set target variable
            X_test: Test set features
            y_test: Test set target variable
            selected_features: A dictionary of selected features
            
        Returns:
            Training results and test results
        """
        logger.info("=" * 50)
        logger.info("STEP 4: Model Training")
        logger.info("=" * 50)
        
        self.model_trainer = ModelTrainer(random_state=self.config['random_state'])
        
        # Use the best feature set (here, using features selected by Mutual Information)
        best_features = selected_features['MutualInfo']
        logger.info(f"Using {len(best_features)} features selected by Mutual Information")
        
        X_train_selected = X_train[best_features]
        X_val_selected = X_val[best_features]
        X_test_selected = X_test[best_features]
        
        # Train all models
        self.training_results = self.model_trainer.train_all_models(
            X_train_selected, y_train, X_val_selected, y_val
        )
        
        logger.info("Training Results:")
        logger.info(self.training_results.to_string())
        
        # Evaluate on the test set
        self.test_results = self.model_trainer.evaluate_on_test_set(X_test_selected, y_test)
        
        logger.info("Test Results:")
        logger.info(self.test_results.to_string())
        
        # Get the best model
        best_model_name, best_model = self.model_trainer.get_best_model('auc')
        
        # Save models and generate reports
        self.model_trainer.save_models()
        self.model_trainer.generate_model_report()
        self.model_trainer.plot_model_comparison()
        
        return self.training_results, self.test_results
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Runs the complete pipeline
        
        Returns:
            A dictionary containing all results
        """
        logger.info("Starting Hospital Readmission Prediction Pipeline")
        logger.info("=" * 60)
        
        try:
            # Step 1: Data Loading
            self.run_data_loading()
            
            # Step 2: Data Preprocessing
            X_train, X_val, X_test, y_train, y_val, y_test = self.run_data_preprocessing()
            
            # Step 3: Feature Selection
            selected_features = self.run_feature_selection(X_train, y_train)
            
            # Step 4: Model Training
            training_results, test_results = self.run_model_training(
                X_train, y_train, X_val, y_val, X_test, y_test, selected_features
            )
            
            # Generate the final report
            self.generate_final_report()
            
            logger.info("Pipeline completed successfully!")
            
            return {
                'raw_data': self.raw_data,
                'processed_data': self.processed_data,
                'selected_features': selected_features,
                'training_results': training_results,
                'test_results': test_results,
                'best_model': self.model_trainer.get_best_model('auc')
            }
            
        except Exception as e:
            logger.error(f"An error occurred during pipeline execution: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}
    
    def generate_final_report(self) -> None:
        """
        Generates a final summary report for the pipeline run.
        """
        logger.info("=" * 50)
        logger.info("STEP 5: Final Report")
        logger.info("=" * 50)

        report = f"""
        =================================================
        HOSPITAL READMISSION PREDICTION PIPELINE REPORT
        =================================================
        
        Date: {pd.to_datetime('today').strftime('%Y-%m-%d %H:%M:%S')}
        
        1. DATA LOADING
        -----------------
        - Raw data shape: {self.raw_data.shape if self.raw_data is not None else 'N/A'}
        
        2. DATA PREPROCESSING
        ---------------------
        - Data shape after processing: {self.processed_data.shape if self.processed_data is not None else 'N/A'}
        
        3. FEATURE SELECTION
        --------------------
        """
        
        if self.selected_features:
            for method, features in self.selected_features.items():
                report += f"- {method}: Selected {len(features)} features.\\n"
        else:
            report += "- No feature selection results available.\\n"
            
        report += """
        4. MODEL TRAINING & EVALUATION
        ------------------------------
        """
        
        if self.test_results is not None:
            report += "Test Set Performance:\\n"
            report += self.test_results.to_string()
        else:
            report += "No model evaluation results available."
            
        logger.info("Final Report:\\n" + report)
        
        # Save the report to a file
        report_path = Path(DATA_PATHS['output_dir']) / 'final_pipeline_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Final report saved to {report_path}")

    def load_and_predict(self, new_data_path: str, model_name: str = None) -> pd.DataFrame:
        """
        Loads a trained model and makes predictions on new data.
        
        Args:
            new_data_path: Path to the new data file (CSV).
            model_name: The name of the model to use. If None, the best model is used.
            
        Returns:
            A DataFrame with predictions.
        """
        logger.info("Loading model and making predictions on new data...")
        
        # Load the model
        if self.model_trainer is None:
            self.model_trainer = ModelTrainer()
            self.model_trainer.load_models()
        
        if model_name is None:
            model_name, model = self.model_trainer.get_best_model('auc')
        else:
            model = self.model_trainer.trained_models.get(model_name)
            if model is None:
                raise ValueError(f"Model '{model_name}' not found.")
        
        logger.info(f"Using model: {model_name}")
        
        # Load and preprocess new data
        new_data = pd.read_csv(new_data_path)
        
        if self.preprocessor is None:
            self.preprocessor = DataPreprocessor()
            # In a real scenario, the preprocessor would need to be saved and loaded
            # to ensure consistent encoding and scaling.
            # For this example, we assume the new data is already somewhat processed.
        
        # This is a simplified preprocessing for prediction
        # A robust implementation would save and load the scaler and encoders.
        
        # Predict
        predictions = model.predict(new_data)
        prediction_probs = model.predict_proba(new_data)[:, 1]
        
        result_df = new_data.copy()
        result_df['prediction'] = predictions
        result_df['prediction_probability'] = prediction_probs
        
        logger.info(f"Predictions completed for {len(new_data)} patients")
        return result_df

def main():
    """Main function to run the pipeline"""
    parser = argparse.ArgumentParser(description="Run the Hospital Readmission Prediction Pipeline.")
    parser.add_argument(
        '--predict', 
        type=str, 
        help="Path to a new data file for prediction."
    )
    parser.add_argument(
        '--model',
        type=str,
        help="Specify the model to use for prediction (e.g., 'LogisticRegression')."
    )
    args = parser.parse_args()
    
    pipeline = HospitalReadmissionPipeline()
    
    if args.predict:
        pipeline.load_and_predict(args.predict, args.model)
    else:
        pipeline.run_full_pipeline()

if __name__ == "__main__":
    main() 