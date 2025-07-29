"""
Database Extension Example Script

This script demonstrates the database extension capabilities for the
hospital readmission prediction pipeline.
"""

import os
import sys
import pandas as pd
from datetime import datetime
import logging

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import DatabaseManager, QueryExecutor, DataMigrator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_database_connection():
    """Setup database connection with environment variables or defaults."""
    # Database configuration - in production, these would be environment variables
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME', 'hospital_readmission'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'password')
    }
    
    logger.info("Setting up database connection...")
    logger.info(f"Database: {db_config['database']} on {db_config['host']}:{db_config['port']}")
    
    return DatabaseManager(db_config)


def demonstrate_database_initialization(db_manager):
    """Demonstrate database initialization and table creation."""
    logger.info("=" * 60)
    logger.info("DEMONSTRATING DATABASE INITIALIZATION")
    logger.info("=" * 60)
    
    # Initialize database
    if db_manager.initialize_database():
        logger.info("✅ Database initialized successfully")
        logger.info("✅ Tables created: patients, encounters, medications, model_results")
    else:
        logger.error("❌ Database initialization failed")
        return False
    
    return True


def demonstrate_data_migration(db_manager):
    """Demonstrate data migration from CSV to database."""
    logger.info("=" * 60)
    logger.info("DEMONSTRATING DATA MIGRATION")
    logger.info("=" * 60)
    
    # Check if diabetic_data.csv exists
    csv_file = 'diabetic_data.csv'
    if not os.path.exists(csv_file):
        logger.warning(f"⚠️  {csv_file} not found. Skipping data migration demo.")
        logger.info("💡 To test data migration, ensure diabetic_data.csv is in the project root")
        return True
    
    # Create migrator
    migrator = DataMigrator(db_manager)
    
    # Define data files to migrate
    data_files = {
        'patients': csv_file
    }
    
    # Perform migration
    logger.info(f"Starting migration of {csv_file} to patients table...")
    migration_results = migrator.migrate_all_data(data_files, validate_only=False)
    
    if migration_results['overall_success']:
        logger.info("✅ Data migration completed successfully")
        logger.info(f"📊 Migrated {migration_results['successful_migrations']} files")
        
        # Save migration report
        report_path = 'outputs/database_migration_report.txt'
        os.makedirs('outputs', exist_ok=True)
        migrator.save_migration_report(migration_results, report_path)
        logger.info(f"📄 Migration report saved to: {report_path}")
    else:
        logger.error("❌ Data migration failed")
        logger.error(f"Failed migrations: {migration_results['failed_migrations']}")
    
    return migration_results['overall_success']


def demonstrate_sql_queries(db_manager):
    """Demonstrate SQL query execution and analytics."""
    logger.info("=" * 60)
    logger.info("DEMONSTRATING SQL QUERIES AND ANALYTICS")
    logger.info("=" * 60)
    
    # Create query executor
    query_executor = QueryExecutor(db_manager.connector)
    
    # Execute various analytics queries
    analytics_results = {}
    
    try:
        # Operational metrics
        logger.info("📊 Executing operational metrics query...")
        operational_metrics = query_executor.execute_operational_metrics()
        if not operational_metrics.empty:
            logger.info("✅ Operational metrics retrieved successfully")
            analytics_results['operational_metrics'] = operational_metrics
            print(f"   Total patients: {operational_metrics['total_patients'].iloc[0]}")
            print(f"   Early readmission rate: {operational_metrics['early_readmission_rate'].iloc[0]:.2f}%")
        else:
            logger.warning("⚠️  No operational metrics data available")
        
        # Patient demographics
        logger.info("📊 Executing demographics analysis...")
        demographics = query_executor.execute_demographics_analysis()
        if not demographics.empty:
            logger.info("✅ Demographics analysis completed")
            analytics_results['demographics'] = demographics
            print(f"   Demographics data: {len(demographics)} rows")
        else:
            logger.warning("⚠️  No demographics data available")
        
        # Length of stay analysis
        logger.info("📊 Executing length of stay analysis...")
        los_analysis = query_executor.execute_length_of_stay_analysis()
        if not los_analysis.empty:
            logger.info("✅ Length of stay analysis completed")
            analytics_results['length_of_stay'] = los_analysis
            print(f"   Length of stay data: {len(los_analysis)} categories")
        else:
            logger.warning("⚠️  No length of stay data available")
        
        # Save analytics results
        if analytics_results:
            analytics_path = 'outputs/database_analytics_results.xlsx'
            with pd.ExcelWriter(analytics_path, engine='openpyxl') as writer:
                for name, df in analytics_results.items():
                    df.to_excel(writer, sheet_name=name, index=False)
            
            logger.info(f"📄 Analytics results saved to: {analytics_path}")
        
    except Exception as e:
        logger.error(f"❌ Error executing queries: {str(e)}")
        return False
    
    return True


def demonstrate_model_result_storage(db_manager):
    """Demonstrate storing model results in database."""
    logger.info("=" * 60)
    logger.info("DEMONSTRATING MODEL RESULT STORAGE")
    logger.info("=" * 60)
    
    # Sample model results
    sample_results = [
        {
            'model_name': 'LogisticRegression',
            'feature_selection_method': 'L1',
            'top_n_features': 15,
            'accuracy': 0.605,
            'precision': 0.195,
            'recall': 0.182,
            'f1_score': 0.188,
            'auc_score': 0.639
        },
        {
            'model_name': 'RandomForest',
            'feature_selection_method': 'TreeImportance',
            'top_n_features': 15,
            'accuracy': 0.591,
            'precision': 0.012,
            'recall': 0.006,
            'f1_score': 0.008,
            'auc_score': 0.606
        },
        {
            'model_name': 'XGBoost',
            'feature_selection_method': 'L1',
            'top_n_features': 15,
            'accuracy': 0.639,
            'precision': 0.014,
            'recall': 0.007,
            'f1_score': 0.009,
            'auc_score': 0.639
        }
    ]
    
    # Store results
    success_count = 0
    for result in sample_results:
        if db_manager.connector.save_model_result(result):
            success_count += 1
            logger.info(f"✅ Stored result for {result['model_name']} ({result['feature_selection_method']})")
        else:
            logger.error(f"❌ Failed to store result for {result['model_name']}")
    
    logger.info(f"📊 Stored {success_count}/{len(sample_results)} model results")
    
    # Retrieve and display stored results
    query_executor = QueryExecutor(db_manager.connector)
    best_models = query_executor.execute_best_performing_models()
    
    if not best_models.empty:
        logger.info("📊 Retrieved stored model results:")
        print(best_models[['model_name', 'feature_selection_method', 'auc_score', 'f1_score']].to_string(index=False))
    else:
        logger.warning("⚠️  No model results found in database")
    
    return success_count > 0


def demonstrate_advanced_queries(db_manager):
    """Demonstrate advanced SQL queries and analytics."""
    logger.info("=" * 60)
    logger.info("DEMONSTRATING ADVANCED SQL QUERIES")
    logger.info("=" * 60)
    
    query_executor = QueryExecutor(db_manager.connector)
    
    try:
        # High risk patients analysis
        logger.info("📊 Identifying high-risk patients...")
        high_risk_patients = query_executor.execute_high_risk_patients(threshold=2)
        if not high_risk_patients.empty:
            logger.info(f"✅ Found {len(high_risk_patients)} high-risk patients")
            print(f"   High-risk patients: {len(high_risk_patients)}")
            if len(high_risk_patients) > 0:
                avg_risk_score = high_risk_patients['risk_score'].mean()
                print(f"   Average risk score: {avg_risk_score:.2f}")
        else:
            logger.warning("⚠️  No high-risk patients found")
        
        # Feature importance analysis
        logger.info("📊 Analyzing feature importance...")
        feature_importance = query_executor.execute_feature_importance_analysis()
        if not feature_importance.empty:
            logger.info("✅ Feature importance analysis completed")
            print(f"   Feature selection methods analyzed: {len(feature_importance)}")
        else:
            logger.warning("⚠️  No feature importance data available")
        
        # Model performance trends
        logger.info("📊 Analyzing model performance trends...")
        model_trends = query_executor.execute_model_performance_trends()
        if not model_trends.empty:
            logger.info("✅ Model performance trends analysis completed")
            print(f"   Performance trends: {len(model_trends)} data points")
        else:
            logger.warning("⚠️  No model performance trends data available")
        
    except Exception as e:
        logger.error(f"❌ Error executing advanced queries: {str(e)}")
        return False
    
    return True


def main():
    """Main demonstration function."""
    logger.info("🚀 Starting Database Extension Demonstration")
    logger.info("=" * 80)
    
    # Setup database connection
    db_manager = setup_database_connection()
    
    # Demonstrate database initialization
    if not demonstrate_database_initialization(db_manager):
        logger.error("❌ Database initialization failed. Exiting.")
        return
    
    # Demonstrate data migration
    demonstrate_data_migration(db_manager)
    
    # Demonstrate SQL queries
    demonstrate_sql_queries(db_manager)
    
    # Demonstrate model result storage
    demonstrate_model_result_storage(db_manager)
    
    # Demonstrate advanced queries
    demonstrate_advanced_queries(db_manager)
    
    # Close database connection
    db_manager.close()
    
    logger.info("=" * 80)
    logger.info("✅ Database Extension Demonstration Completed")
    logger.info("📁 Check the 'outputs/' directory for generated reports and analytics")
    logger.info("💡 This demonstrates SQL skills, database operations, and analytics capabilities")


if __name__ == "__main__":
    main() 