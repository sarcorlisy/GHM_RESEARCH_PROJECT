"""
API Configuration for Hospital Readmission Project
Defines API endpoints, authentication, and rate limiting
"""

import os
from typing import Dict, Any, Optional
import yaml

class APIConfig:
    """Configuration manager for external APIs"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/api_config.yaml"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load API configuration from YAML file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as file:
                    return yaml.safe_load(file)
            else:
                return self._get_default_config()
        except Exception as e:
            print(f"Warning: Could not load API config from {self.config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default API configuration"""
        return {
            "apis": {
                "icd10": {
                    "base_url": "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search",
                    "rate_limit": 100,  # requests per minute
                    "timeout": 30,
                    "enabled": True
                },
                "openfda": {
                    "base_url": "https://api.fda.gov/drug",
                    "rate_limit": 1000,  # requests per day
                    "timeout": 30,
                    "enabled": True
                },
                "fhir_simulated": {
                    "base_url": "http://localhost:8000/fhir",  # Simulated FHIR endpoint
                    "rate_limit": 1000,
                    "timeout": 10,
                    "enabled": True
                },
                "data_quality": {
                    "base_url": "https://api.dataquality.com",  # Simulated data quality API
                    "rate_limit": 500,
                    "timeout": 15,
                    "enabled": True
                }
            },
            "lerner_research": {
                "base_url": "https://api.lerner.ccf.org",  # Simulated Lerner Research Institute API
                "rate_limit": 200,
                "timeout": 20,
                "enabled": True,
                "endpoints": {
                    "patient_demographics": "/api/v1/patients/demographics",
                    "clinical_notes": "/api/v1/patients/notes",
                    "lab_results": "/api/v1/patients/labs"
                }
            },
            "external_vendors": {
                "base_url": "https://api.externalvendor.com",  # Simulated external vendor API
                "rate_limit": 300,
                "timeout": 25,
                "enabled": True,
                "endpoints": {
                    "medication_info": "/api/v1/medications",
                    "insurance_data": "/api/v1/insurance",
                    "provider_info": "/api/v1/providers"
                }
            },
            "research_teams": {
                "base_url": "https://api.research.ccf.org",  # Simulated research operational teams API
                "rate_limit": 150,
                "timeout": 15,
                "enabled": True,
                "endpoints": {
                    "study_participants": "/api/v1/studies/participants",
                    "protocol_data": "/api/v1/studies/protocols",
                    "outcome_measures": "/api/v1/studies/outcomes"
                }
            }
        }
    
    def get_api_config(self, api_name: str) -> Dict[str, Any]:
        """Get configuration for specific API"""
        return self.config.get("apis", {}).get(api_name, {})
    
    def get_lerner_config(self) -> Dict[str, Any]:
        """Get Lerner Research Institute API configuration"""
        return self.config.get("lerner_research", {})
    
    def get_vendor_config(self) -> Dict[str, Any]:
        """Get external vendor API configuration"""
        return self.config.get("external_vendors", {})
    
    def get_research_config(self) -> Dict[str, Any]:
        """Get research teams API configuration"""
        return self.config.get("research_teams", {})
    
    def get_sampling_config(self) -> Dict[str, Any]:
        """Get sampling configuration"""
        return self.config.get("sampling", {})
    
    def is_api_enabled(self, api_name: str) -> bool:
        """Check if specific API is enabled"""
        api_config = self.get_api_config(api_name)
        return api_config.get("enabled", False)
    
    def get_rate_limit(self, api_name: str) -> int:
        """Get rate limit for specific API"""
        api_config = self.get_api_config(api_name)
        return api_config.get("rate_limit", 100)
    
    def get_timeout(self, api_name: str) -> int:
        """Get timeout for specific API"""
        api_config = self.get_api_config(api_name)
        return api_config.get("timeout", 30) 