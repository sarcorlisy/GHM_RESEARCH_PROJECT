#!/usr/bin/env python3
"""
API Integration Runner for Hospital Readmission Project
Demonstrates heterogeneous data source integration capabilities
"""

import argparse
import logging
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from api_integration.data_enricher import DataEnricher
from api_integration.api_manager import APIManager
from api_integration.api_config import APIConfig
from utils.config import ConfigManager
from utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

def setup_logging_config():
    """Setup logging configuration"""
    setup_logging(
        log_level="INFO",
        log_file="logs/api_integration.log",
        log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def test_api_connections():
    """Test API connections and configurations"""
    logger.info("🔧 Testing API connections...")
    
    api_manager = APIManager()
    api_config = APIConfig()
    
    # Test ICD-10 API
    logger.info("Testing ICD-10 API...")
    response = api_manager.call_icd10_api("E11.9")
    if response.success:
        logger.info("✅ ICD-10 API connection successful")
        logger.info(f"   Response: {response.data}")
    else:
        logger.warning(f"⚠️ ICD-10 API connection failed: {response.error_message}")
    
    # Test OpenFDA API
    logger.info("Testing OpenFDA API...")
    response = api_manager.call_openfda_api("metformin")
    if response.success:
        logger.info("✅ OpenFDA API connection successful")
        logger.info(f"   Response: {response.data}")
    else:
        logger.warning(f"⚠️ OpenFDA API connection failed: {response.error_message}")
    
    # Test FHIR Simulated API
    logger.info("Testing FHIR Simulated API...")
    response = api_manager.call_fhir_simulated_api("12345")
    if response.success:
        logger.info("✅ FHIR Simulated API connection successful")
        logger.info(f"   Response: {response.data}")
    else:
        logger.warning(f"⚠️ FHIR Simulated API connection failed: {response.error_message}")
    
    # Test Data Quality API
    logger.info("Testing Data Quality API...")
    test_data = {"patient_id": "12345", "age": 65, "gender": "Female"}
    response = api_manager.call_data_quality_api(test_data)
    if response.success:
        logger.info("✅ Data Quality API connection successful")
        logger.info(f"   Response: {response.data}")
    else:
        logger.warning(f"⚠️ Data Quality API connection failed: {response.error_message}")

def run_diagnosis_enrichment():
    """Run diagnosis data enrichment only"""
    logger.info("🔍 Running diagnosis data enrichment...")
    
    enricher = DataEnricher()
    
    # Download cleaned data
    df = enricher.download_cleaned_data_from_azure()
    if df is None:
        logger.error("❌ Failed to download cleaned data")
        return False
    
    # Enrich diagnosis data only
    df_enriched = enricher.enrich_diagnosis_data(df)
    
    # Save to MySQL
    if enricher.connect_to_mysql():
        success = enricher.save_enriched_data_to_mysql(
            df_enriched, 
            table_name="patients_diagnosis_enriched"
        )
        enricher.disconnect_from_mysql()
        
        if success:
            logger.info("✅ Diagnosis enrichment completed successfully")
            return True
        else:
            logger.error("❌ Failed to save diagnosis enriched data")
            return False
    else:
        logger.error("❌ Failed to connect to MySQL")
        return False

def run_medication_enrichment():
    """Run medication data enrichment only"""
    logger.info("💊 Running medication data enrichment...")
    
    enricher = DataEnricher()
    
    # Download cleaned data
    df = enricher.download_cleaned_data_from_azure()
    if df is None:
        logger.error("❌ Failed to download cleaned data")
        return False
    
    # Enrich medication data only
    df_enriched = enricher.enrich_medication_data(df)
    
    # Save to MySQL
    if enricher.connect_to_mysql():
        success = enricher.save_enriched_data_to_mysql(
            df_enriched, 
            table_name="patients_medication_enriched"
        )
        enricher.disconnect_from_mysql()
        
        if success:
            logger.info("✅ Medication enrichment completed successfully")
            return True
        else:
            logger.error("❌ Failed to save medication enriched data")
            return False
    else:
        logger.error("❌ Failed to connect to MySQL")
        return False

def run_heterogeneous_enrichment():
    """Run heterogeneous data source enrichment"""
    logger.info("🌐 Running heterogeneous data source enrichment...")
    
    enricher = DataEnricher()
    
    # Download cleaned data
    df = enricher.download_cleaned_data_from_azure()
    if df is None:
        logger.error("❌ Failed to download cleaned data")
        return False
    
    # Enrich with heterogeneous sources
    df_enriched = enricher.enrich_with_heterogeneous_sources(df)
    
    # Save to MySQL
    if enricher.connect_to_mysql():
        success = enricher.save_enriched_data_to_mysql(
            df_enriched, 
            table_name="patients_heterogeneous_enriched"
        )
        enricher.disconnect_from_mysql()
        
        if success:
            logger.info("✅ Heterogeneous enrichment completed successfully")
            return True
        else:
            logger.error("❌ Failed to save heterogeneous enriched data")
            return False
    else:
        logger.error("❌ Failed to connect to MySQL")
        return False

def run_full_enrichment():
    """Run complete data enrichment process"""
    logger.info("🚀 Running complete data enrichment process...")
    
    enricher = DataEnricher()
    success = enricher.run_full_enrichment()
    
    if success:
        logger.info("🎉 Complete data enrichment process finished successfully!")
        return True
    else:
        logger.error("❌ Complete data enrichment process failed")
        return False

def generate_enrichment_report():
    """Generate a report of the enrichment process"""
    logger.info("📊 Generating enrichment report...")
    
    enricher = DataEnricher()
    
    # Download cleaned data for analysis
    df = enricher.download_cleaned_data_from_azure()
    if df is None:
        logger.error("❌ Failed to download cleaned data for report")
        return False
    
    # Generate quality report
    quality_summary = enricher.validate_data_quality(df)
    
    # Create report
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_records": len(df),
        "data_quality_score": quality_summary['average_quality_score'],
        "total_issues": quality_summary['total_issues_found'],
        "common_issues": quality_summary['common_issues'],
        "recommendations": quality_summary['recommendations'],
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": df.dtypes.to_dict()
    }
    
    # Save report
    import json
    report_file = "logs/enrichment_report.json"
    os.makedirs("logs", exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Enrichment report saved to {report_file}")
    logger.info(f"📈 Data Quality Score: {report['data_quality_score']:.2f}")
    logger.info(f"📊 Total Records: {report['total_records']}")
    logger.info(f"⚠️ Total Issues: {report['total_issues']}")
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="API Integration for Hospital Readmission Project")
    parser.add_argument("--test", action="store_true", help="Test API connections")
    parser.add_argument("--diagnosis", action="store_true", help="Run diagnosis enrichment only")
    parser.add_argument("--medication", action="store_true", help="Run medication enrichment only")
    parser.add_argument("--heterogeneous", action="store_true", help="Run heterogeneous enrichment only")
    parser.add_argument("--full", action="store_true", help="Run complete enrichment process")
    parser.add_argument("--report", action="store_true", help="Generate enrichment report")
    parser.add_argument("--all", action="store_true", help="Run all enrichment processes")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging_config()
    
    logger.info("🚀 Starting API Integration for Hospital Readmission Project")
    logger.info("=" * 60)
    
    success_count = 0
    total_tests = 0
    
    try:
        # Test API connections
        if args.test or args.all:
            total_tests += 1
            if test_api_connections():
                success_count += 1
        
        # Run diagnosis enrichment
        if args.diagnosis or args.all:
            total_tests += 1
            if run_diagnosis_enrichment():
                success_count += 1
        
        # Run medication enrichment
        if args.medication or args.all:
            total_tests += 1
            if run_medication_enrichment():
                success_count += 1
        
        # Run heterogeneous enrichment
        if args.heterogeneous or args.all:
            total_tests += 1
            if run_heterogeneous_enrichment():
                success_count += 1
        
        # Run full enrichment
        if args.full or args.all:
            total_tests += 1
            if run_full_enrichment():
                success_count += 1
        
        # Generate report
        if args.report or args.all:
            total_tests += 1
            if generate_enrichment_report():
                success_count += 1
        
        # If no specific option is provided, run tests
        if not any([args.test, args.diagnosis, args.medication, args.heterogeneous, args.full, args.report, args.all]):
            logger.info("No specific option provided. Running API connection tests...")
            total_tests += 1
            if test_api_connections():
                success_count += 1
        
        # Summary
        logger.info("=" * 60)
        logger.info(f"📊 Summary: {success_count}/{total_tests} tests completed successfully")
        
        if success_count == total_tests:
            logger.info("🎉 All tests passed! API integration is working correctly.")
            return 0
        else:
            logger.warning(f"⚠️ {total_tests - success_count} tests failed. Check logs for details.")
            return 1
            
    except KeyboardInterrupt:
        logger.info("⏹️ Process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 