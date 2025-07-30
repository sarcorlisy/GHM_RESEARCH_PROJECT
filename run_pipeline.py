#!/usr/bin/env python3
"""
Hospital Readmission Prediction Pipeline - Main Runner
Enterprise-grade data processing pipeline with Azure integration
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.config import ConfigManager
from src.utils.logging_config import setup_logging
from src.etl.etl_pipeline import HospitalReadmissionETL
from src.data_ingestion.azure_uploader import AzureDataUploader

def main():
    """Main pipeline runner"""
    parser = argparse.ArgumentParser(description="Hospital Readmission ETL Pipeline")
    parser.add_argument("--config", default="development", choices=["development", "production"],
                       help="Configuration environment")
    parser.add_argument("--upload-azure", action="store_true", default=False,
                       help="Upload data to Azure Data Lake")
    parser.add_argument("--import-azure", action="store_true", default=True,
                       help="Import data from Azure (default: True)")
    parser.add_argument("--step", choices=["cleaning", "features", "full"],
                       default="full", help="Pipeline step to run")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(
        log_level=args.log_level,
        log_file=f"logs/pipeline_{args.config}.log"
    )
    
    logger.info("🚀 Starting Hospital Readmission ETL Pipeline")
    logger.info(f"📋 Configuration: {args.config}")
    logger.info(f"📤 Upload to Azure: {args.upload_azure}")
    logger.info(f"📥 Import from Azure: {args.import_azure}")
    logger.info(f"🔧 Pipeline step: {args.step}")
    
    try:
        # Initialize configuration
        config = ConfigManager()
        
        # Initialize ETL pipeline
        etl = HospitalReadmissionETL(config)
        
        # Run pipeline based on step
        if args.step == "cleaning":
            logger.info("🧹 Running data cleaning only...")
            if etl.connect_to_mysql():
                success = etl.run_data_cleaning()
                etl.disconnect_from_mysql()
                return 0 if success else 1
            else:
                return 1
                
        elif args.step == "features":
            logger.info("🔧 Running feature engineering only...")
            if etl.connect_to_mysql():
                success = etl.run_feature_engineering()
                etl.disconnect_from_mysql()
                return 0 if success else 1
            else:
                return 1
                
        elif args.step == "full":
            logger.info("🎯 Running full ETL pipeline...")
            success = etl.run_full_pipeline(
                upload_to_azure=args.upload_azure,
                import_from_azure=args.import_azure
            )
            return 0 if success else 1
            
    except KeyboardInterrupt:
        logger.info("⏹️ Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Pipeline failed with error: {e}")
        return 1

def run_azure_upload():
    """Run Azure upload only"""
    logger = setup_logging(log_level="INFO")
    logger.info("☁️ Starting Azure upload...")
    
    try:
        config = ConfigManager()
        azure_uploader = AzureDataUploader(config)
        
        # Upload original CSV file
        if os.path.exists("diabetic_data.csv"):
            success = azure_uploader.upload_csv_to_azure(
                local_file_path="diabetic_data.csv",
                container_name="raw-data",
                blob_name="diabetic_data_raw.csv"
            )
            if success:
                logger.info("✅ Successfully uploaded raw data to Azure")
            else:
                logger.warning("⚠️ Failed to upload raw data to Azure")
        
        # List blobs in container
        blobs = azure_uploader.list_blobs_in_container("raw-data")
        if blobs:
            logger.info(f"📋 Found {len(blobs)} blobs in raw-data container")
            for blob in blobs:
                logger.info(f"  - {blob}")
        
    except Exception as e:
        logger.error(f"❌ Azure upload failed: {e}")

if __name__ == "__main__":
    # Check if this is an Azure upload request
    if len(sys.argv) > 1 and sys.argv[1] == "azure-upload":
        run_azure_upload()
    else:
        exit_code = main()
        sys.exit(exit_code) 