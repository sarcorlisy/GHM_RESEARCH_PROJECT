"""
API Manager for Hospital Readmission Project
Handles API calls, rate limiting, and error handling
"""

import time
import requests
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from .api_config import APIConfig

logger = logging.getLogger(__name__)

@dataclass
class APIResponse:
    """Standardized API response"""
    success: bool
    data: Any
    status_code: int
    error_message: Optional[str] = None
    api_name: Optional[str] = None
    timestamp: Optional[datetime] = None

class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, max_requests: int, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window  # seconds
        self.requests = []
    
    def can_make_request(self) -> bool:
        """Check if we can make a request"""
        now = datetime.now()
        # Remove old requests outside the time window
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < timedelta(seconds=self.time_window)]
        
        return len(self.requests) < self.max_requests
    
    def record_request(self):
        """Record a request"""
        self.requests.append(datetime.now())
    
    def wait_if_needed(self):
        """Wait if rate limit is reached"""
        while not self.can_make_request():
            time.sleep(1)

class APIManager:
    """Manages API calls with rate limiting and error handling"""
    
    def __init__(self, config: Optional[APIConfig] = None):
        self.config = config or APIConfig()
        self.rate_limiters = {}
        self.session = requests.Session()
        self._setup_rate_limiters()
    
    def _setup_rate_limiters(self):
        """Setup rate limiters for each API"""
        for api_name in ["icd9", "icd10", "openfda", "fhir_simulated", "data_quality"]:
            if self.config.is_api_enabled(api_name):
                rate_limit = self.config.get_rate_limit(api_name)
                self.rate_limiters[api_name] = RateLimiter(rate_limit)
    
    def _get_rate_limiter(self, api_name: str) -> Optional[RateLimiter]:
        """Get rate limiter for specific API"""
        return self.rate_limiters.get(api_name)
    
    def _make_request(self, api_name: str, url: str, params: Optional[Dict] = None, 
                     headers: Optional[Dict] = None, method: str = "GET") -> APIResponse:
        """Make API request with rate limiting and error handling"""
        rate_limiter = self._get_rate_limiter(api_name)
        
        if rate_limiter:
            rate_limiter.wait_if_needed()
        
        try:
            timeout = self.config.get_timeout(api_name)
            
            if method.upper() == "GET":
                response = self.session.get(url, params=params, headers=headers, timeout=timeout)
            elif method.upper() == "POST":
                response = self.session.post(url, json=params, headers=headers, timeout=timeout)
            else:
                return APIResponse(
                    success=False,
                    data=None,
                    status_code=400,
                    error_message=f"Unsupported HTTP method: {method}",
                    api_name=api_name,
                    timestamp=datetime.now()
                )
            
            if rate_limiter:
                rate_limiter.record_request()
            
            if response.status_code == 200:
                try:
                    data = response.json()
                except json.JSONDecodeError:
                    data = response.text
                
                return APIResponse(
                    success=True,
                    data=data,
                    status_code=response.status_code,
                    api_name=api_name,
                    timestamp=datetime.now()
                )
            else:
                return APIResponse(
                    success=False,
                    data=None,
                    status_code=response.status_code,
                    error_message=f"API request failed: {response.text}",
                    api_name=api_name,
                    timestamp=datetime.now()
                )
                
        except requests.exceptions.Timeout:
            return APIResponse(
                success=False,
                data=None,
                status_code=408,
                error_message="Request timeout",
                api_name=api_name,
                timestamp=datetime.now()
            )
        except requests.exceptions.RequestException as e:
            return APIResponse(
                success=False,
                data=None,
                status_code=500,
                error_message=f"Request failed: {str(e)}",
                api_name=api_name,
                timestamp=datetime.now()
            )
    
    def call_icd9_api(self, diagnosis_code: str) -> APIResponse:
        """Call ICD-9 API for diagnosis information"""
        if not self.config.is_api_enabled("icd9"):
            return APIResponse(
                success=False,
                data=None,
                status_code=400,
                error_message="ICD-9 API is disabled",
                api_name="icd9",
                timestamp=datetime.now()
            )
        
        config = self.config.get_api_config("icd9")
        url = config["base_url"]
        params = {
            "terms": diagnosis_code,
            "maxList": 1
        }
        
        return self._make_request("icd9", url, params=params)
    
    def call_icd10_api(self, diagnosis_code: str) -> APIResponse:
        """Call ICD-10 API for diagnosis information"""
        if not self.config.is_api_enabled("icd10"):
            return APIResponse(
                success=False,
                data=None,
                status_code=400,
                error_message="ICD-10 API is disabled",
                api_name="icd10",
                timestamp=datetime.now()
            )
        
        config = self.config.get_api_config("icd10")
        url = config["base_url"]
        params = {
            "terms": diagnosis_code,
            "maxList": 1
        }
        
        return self._make_request("icd10", url, params=params)
    
    def call_openfda_api(self, drug_name: str) -> APIResponse:
        """Call OpenFDA API for drug information"""
        if not self.config.is_api_enabled("openfda"):
            return APIResponse(
                success=False,
                data=None,
                status_code=400,
                error_message="OpenFDA API is disabled",
                api_name="openfda",
                timestamp=datetime.now()
            )
        
        config = self.config.get_api_config("openfda")
        url = f"{config['base_url']}/label.json"
        params = {
            "search": f"openfda.generic_name:{drug_name}",
            "limit": 1
        }
        
        return self._make_request("openfda", url, params=params)
    
    def call_fhir_simulated_api(self, patient_id: str) -> APIResponse:
        """Call simulated FHIR API for patient data"""
        if not self.config.is_api_enabled("fhir_simulated"):
            return APIResponse(
                success=False,
                data=None,
                status_code=400,
                error_message="FHIR API is disabled",
                api_name="fhir_simulated",
                timestamp=datetime.now()
            )
        
        # Simulated FHIR response
        simulated_data = {
            "resourceType": "Patient",
            "id": patient_id,
            "active": True,
            "gender": "unknown",
            "birthDate": "1980-01-01",
            "deceasedBoolean": False,
            "address": [
                {
                    "use": "home",
                    "type": "physical",
                    "text": "123 Main St, Cleveland, OH 44195",
                    "city": "Cleveland",
                    "state": "OH",
                    "postalCode": "44195"
                }
            ],
            "contact": [
                {
                    "relationship": [
                        {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/v2-0131",
                                    "code": "C",
                                    "display": "Emergency Contact"
                                }
                            ]
                        }
                    ],
                    "telecom": [
                        {
                            "system": "phone",
                            "value": "555-123-4567"
                        }
                    ]
                }
            ]
        }
        
        return APIResponse(
            success=True,
            data=simulated_data,
            status_code=200,
            api_name="fhir_simulated",
            timestamp=datetime.now()
        )
    
    def call_data_quality_api(self, data_sample: Dict[str, Any]) -> APIResponse:
        """Call data quality API for data validation"""
        if not self.config.is_api_enabled("data_quality"):
            return APIResponse(
                success=False,
                data=None,
                status_code=400,
                error_message="Data Quality API is disabled",
                api_name="data_quality",
                timestamp=datetime.now()
            )
        
        # Simulated data quality response
        quality_score = 0.85  # Simulated quality score
        issues = []
        
        # Simulate data quality checks
        for field, value in data_sample.items():
            if value is None or value == "":
                issues.append(f"Missing value in field: {field}")
            elif isinstance(value, str) and len(value) > 100:
                issues.append(f"Field {field} exceeds maximum length")
        
        quality_data = {
            "quality_score": quality_score,
            "issues": issues,
            "recommendations": [
                "Consider imputing missing values",
                "Validate data ranges",
                "Check for outliers"
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        return APIResponse(
            success=True,
            data=quality_data,
            status_code=200,
            api_name="data_quality",
            timestamp=datetime.now()
        )
    
    def call_lerner_research_api(self, endpoint: str, params: Optional[Dict] = None) -> APIResponse:
        """Call Lerner Research Institute API"""
        config = self.config.get_lerner_config()
        if not config.get("enabled", False):
            return APIResponse(
                success=False,
                data=None,
                status_code=400,
                error_message="Lerner Research API is disabled",
                api_name="lerner_research",
                timestamp=datetime.now()
            )
        
        # Simulated Lerner Research Institute data
        patient_id = params.get("patient_id", "unknown") if params else "unknown"
        
        simulated_data = {
            "patient_id": patient_id,
            "demographics": {
                "age_group": "65-75",
                "education_level": "Bachelor's Degree",
                "employment_status": "Retired",
                "household_income": "50k-75k",
                "marital_status": "Married"
            },
            "clinical_notes": [
                "Patient shows good compliance with medication regimen",
                "Regular follow-up appointments maintained",
                "Blood pressure well controlled"
            ],
            "lab_results": {
                "a1c": "7.2%",
                "creatinine": "1.1 mg/dL",
                "egfr": "65 mL/min/1.73m²"
            },
            "research_participation": {
                "enrolled_studies": ["DIABETES_CARE_2024", "CARDIOVASCULAR_OUTCOMES"],
                "consent_date": "2024-01-15",
                "last_visit": "2024-07-20"
            }
        }
        
        return APIResponse(
            success=True,
            data=simulated_data,
            status_code=200,
            api_name="lerner_research",
            timestamp=datetime.now()
        )
    
    def call_external_vendor_api(self, endpoint: str, params: Optional[Dict] = None) -> APIResponse:
        """Call external vendor API"""
        config = self.config.get_vendor_config()
        if not config.get("enabled", False):
            return APIResponse(
                success=False,
                data=None,
                status_code=400,
                error_message="External Vendor API is disabled",
                api_name="external_vendors",
                timestamp=datetime.now()
            )
        
        # Simulated External Vendor data
        patient_id = params.get("patient_id", "unknown") if params else "unknown"
        
        simulated_data = {
            "patient_id": patient_id,
            "insurance": {
                "provider": "Blue Cross Blue Shield",
                "plan_type": "PPO",
                "member_id": f"BCBS{patient_id[-6:]}",
                "group_number": "123456",
                "effective_date": "2024-01-01",
                "copay": {
                    "primary_care": "$25",
                    "specialist": "$40",
                    "emergency": "$150"
                }
            },
            "medication_coverage": {
                "metformin": "Covered - Tier 1",
                "insulin": "Covered - Tier 2",
                "glipizide": "Covered - Tier 1",
                "prior_authorization_required": False
            },
            "provider_network": {
                "primary_care": "Dr. Sarah Johnson",
                "endocrinologist": "Dr. Michael Chen",
                "pharmacy": "CVS Pharmacy"
            }
        }
        
        return APIResponse(
            success=True,
            data=simulated_data,
            status_code=200,
            api_name="external_vendors",
            timestamp=datetime.now()
        )
    
    def call_research_teams_api(self, endpoint: str, params: Optional[Dict] = None) -> APIResponse:
        """Call research operational teams API"""
        config = self.config.get_research_config()
        if not config.get("enabled", False):
            return APIResponse(
                success=False,
                data=None,
                status_code=400,
                error_message="Research Teams API is disabled",
                api_name="research_teams",
                timestamp=datetime.now()
            )
        
        # Simulated Research Teams data
        patient_id = params.get("patient_id", "unknown") if params else "unknown"
        
        simulated_data = {
            "patient_id": patient_id,
            "study_participation": {
                "enrolled": True,
                "study_id": "DIABETES_CARE_2024",
                "protocol_version": "2.1",
                "enrollment_date": "2024-02-15",
                "consent_status": "Signed"
            },
            "protocol_data": {
                "intervention_group": "Standard Care + Telemedicine",
                "randomization_date": "2024-02-20",
                "baseline_visit": "2024-02-25",
                "follow_up_schedule": "Monthly"
            },
            "outcome_measures": {
                "primary_endpoint": "HbA1c reduction",
                "secondary_endpoints": ["Blood pressure", "Weight loss", "Quality of life"],
                "baseline_a1c": "8.5%",
                "current_a1c": "7.8%"
            },
            "compliance": {
                "medication_adherence": "95%",
                "visit_attendance": "100%",
                "data_completion": "98%",
                "last_assessment": "2024-07-15"
            }
        }
        
        return APIResponse(
            success=True,
            data=simulated_data,
            status_code=200,
            api_name="research_teams",
            timestamp=datetime.now()
        ) 