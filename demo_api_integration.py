#!/usr/bin/env python3
"""
API Integration Demo for Hospital Readmission Project
Simple demonstration of the API integration capabilities
"""

import sys
import os
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import API modules
try:
    from api_integration.api_manager import APIManager
    from api_integration.api_config import APIConfig
except ImportError:
    print("❌ Import error. Please ensure you're running from the project root directory.")
    sys.exit(1)

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_icd9_api():
    """Demonstrate ICD-9 API functionality"""
    print("\n" + "="*50)
    print("🔍 ICD-9 API Demo")
    print("="*50)
    
    api_manager = APIManager()
    
    # Test with common diabetes diagnosis codes from the dataset
    test_codes = ["250", "250.7", "250.43", "250.6"]
    
    for code in test_codes:
        print(f"\n📋 Testing ICD-9 code: {code}")
        response = api_manager.call_icd9_api(code)
        
        if response.success:
            print(f"✅ Success! Found diagnosis information")
            if isinstance(response.data, dict) and 'results' in response.data:
                results = response.data['results']
                if results:
                    result = results[0]
                    print(f"   Name: {result.get('name', 'N/A')}")
                    print(f"   Category: {result.get('category', 'N/A')}")
        else:
            print(f"❌ Failed: {response.error_message}")

def demo_openfda_api():
    """Demonstrate OpenFDA API functionality"""
    print("\n" + "="*50)
    print("💊 OpenFDA API Demo")
    print("="*50)
    
    api_manager = APIManager()
    
    # Test with common diabetes medications
    test_medications = ["metformin", "glipizide", "insulin"]
    
    for med in test_medications:
        print(f"\n💊 Testing medication: {med}")
        response = api_manager.call_openfda_api(med)
        
        if response.success:
            print(f"✅ Success! Found medication information")
            if isinstance(response.data, dict) and 'results' in response.data:
                results = response.data['results']
                if results:
                    result = results[0]
                    openfda = result.get('openfda', {})
                    print(f"   Generic Name: {openfda.get('generic_name', ['N/A'])[0] if openfda.get('generic_name') else 'N/A'}")
                    print(f"   Brand Name: {openfda.get('brand_name', ['N/A'])[0] if openfda.get('brand_name') else 'N/A'}")
        else:
            print(f"❌ Failed: {response.error_message}")

def demo_fhir_api():
    """Demonstrate FHIR API functionality"""
    print("\n" + "="*50)
    print("👤 FHIR API Demo")
    print("="*50)
    
    api_manager = APIManager()
    
    # Test with sample patient IDs
    test_patients = ["12345", "67890", "11111"]
    
    for patient_id in test_patients:
        print(f"\n👤 Testing patient ID: {patient_id}")
        response = api_manager.call_fhir_simulated_api(patient_id)
        
        if response.success:
            print(f"✅ Success! Found patient information")
            fhir_data = response.data
            print(f"   Active: {fhir_data.get('active', 'N/A')}")
            print(f"   Gender: {fhir_data.get('gender', 'N/A')}")
            print(f"   Birth Date: {fhir_data.get('birthDate', 'N/A')}")
            
            # Show address info
            if fhir_data.get('address') and len(fhir_data['address']) > 0:
                address = fhir_data['address'][0]
                print(f"   City: {address.get('city', 'N/A')}")
                print(f"   State: {address.get('state', 'N/A')}")
        else:
            print(f"❌ Failed: {response.error_message}")

def demo_data_quality_api():
    """Demonstrate Data Quality API functionality"""
    print("\n" + "="*50)
    print("🔍 Data Quality API Demo")
    print("="*50)
    
    api_manager = APIManager()
    
    # Test with sample patient data
    test_records = [
        {"patient_id": "12345", "age": 65, "gender": "Female", "diag_1": "E11.9"},
        {"patient_id": "67890", "age": None, "gender": "", "diag_1": "E10.9"},
        {"patient_id": "11111", "age": 45, "gender": "Male", "diag_1": "E13.9"}
    ]
    
    for i, record in enumerate(test_records, 1):
        print(f"\n📊 Testing data quality for record {i}")
        response = api_manager.call_data_quality_api(record)
        
        if response.success:
            print(f"✅ Success! Quality assessment completed")
            quality_data = response.data
            print(f"   Quality Score: {quality_data.get('quality_score', 'N/A')}")
            print(f"   Issues Found: {len(quality_data.get('issues', []))}")
            if quality_data.get('issues'):
                print(f"   Issues: {', '.join(quality_data['issues'])}")
        else:
            print(f"❌ Failed: {response.error_message}")

def demo_heterogeneous_apis():
    """Demonstrate heterogeneous data source APIs"""
    print("\n" + "="*50)
    print("🌐 Heterogeneous Data Sources Demo")
    print("="*50)
    
    api_manager = APIManager()
    
    # Test Lerner Research Institute API
    print("\n🏥 Testing Lerner Research Institute API...")
    lerner_response = api_manager.call_lerner_research_api(
        "/api/v1/patients/demographics",
        {"patient_id": "12345"}
    )
    if lerner_response.success:
        print("✅ Lerner Research API: Success")
    else:
        print(f"❌ Lerner Research API: {lerner_response.error_message}")
    
    # Test External Vendor API
    print("\n🏢 Testing External Vendor API...")
    vendor_response = api_manager.call_external_vendor_api(
        "/api/v1/insurance",
        {"patient_id": "12345"}
    )
    if vendor_response.success:
        print("✅ External Vendor API: Success")
    else:
        print(f"❌ External Vendor API: {vendor_response.error_message}")
    
    # Test Research Teams API
    print("\n🔬 Testing Research Teams API...")
    research_response = api_manager.call_research_teams_api(
        "/api/v1/studies/participants",
        {"patient_id": "12345"}
    )
    if research_response.success:
        print("✅ Research Teams API: Success")
    else:
        print(f"❌ Research Teams API: {research_response.error_message}")

def main():
    """Main demo function"""
    print("🚀 Hospital Readmission Project - API Integration Demo")
    print("="*60)
    print(f"📅 Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 This demo showcases heterogeneous data source integration capabilities")
    
    try:
        # Run all demos
        demo_icd9_api()
        demo_openfda_api()
        demo_fhir_api()
        demo_data_quality_api()
        demo_heterogeneous_apis()
        
        print("\n" + "="*60)
        print("🎉 API Integration Demo Completed Successfully!")
        print("="*60)
        print("\n📋 Summary of demonstrated capabilities:")
        print("   ✅ ICD-9 API for diagnosis enrichment")
        print("   ✅ OpenFDA API for medication information")
        print("   ✅ FHIR API for patient demographic data")
        print("   ✅ Data Quality API for validation")
        print("   ✅ Heterogeneous data source integration")
        print("      - Lerner Research Institute")
        print("      - External Vendors")
        print("      - Research Operational Teams")
        
        print("\n💡 Next steps:")
        print("   1. Run 'python run_api_integration.py --test' to test all APIs")
        print("   2. Run 'python run_api_integration.py --full' for complete enrichment")
        print("   3. Check logs/api_integration.log for detailed information")
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        logger.error(f"Demo error: {e}")

if __name__ == "__main__":
    main() 