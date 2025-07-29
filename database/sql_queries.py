"""
SQL Queries Module for Hospital Readmission Prediction Pipeline

This module contains predefined SQL queries for common analytics operations
in the hospital readmission prediction workflow.
"""

from typing import Dict, List, Optional
import pandas as pd


class HospitalReadmissionQueries:
    """
    Collection of SQL queries for hospital readmission analysis.
    
    This class provides pre-defined queries for:
    - Patient demographics analysis
    - Readmission patterns
    - Clinical indicators
    - Model performance tracking
    - Operational metrics
    """
    
    @staticmethod
    def get_patient_demographics() -> str:
        """Get patient demographics summary."""
        return """
        SELECT 
            age,
            gender,
            race,
            COUNT(*) as patient_count,
            AVG(time_in_hospital) as avg_length_of_stay,
            SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) as early_readmissions,
            SUM(CASE WHEN readmitted = '>30' THEN 1 ELSE 0 END) as late_readmissions,
            SUM(CASE WHEN readmitted = 'NO' THEN 1 ELSE 0 END) as no_readmissions,
            ROUND(
                SUM(CASE WHEN readmitted IN ('<30', '>30') THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
            ) as overall_readmission_rate
        FROM patients 
        GROUP BY age, gender, race
        ORDER BY age, gender, race
        """
    
    @staticmethod
    def get_readmission_risk_factors() -> str:
        """Get readmission risk factors analysis."""
        return """
        SELECT 
            admission_type_id,
            discharge_disposition_id,
            admission_source_id,
            COUNT(*) as total_cases,
            SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) as early_readmissions,
            ROUND(
                SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
            ) as early_readmission_rate,
            AVG(time_in_hospital) as avg_length_of_stay,
            AVG(num_medications) as avg_medications,
            AVG(num_lab_procedures) as avg_lab_procedures
        FROM patients 
        GROUP BY admission_type_id, discharge_disposition_id, admission_source_id
        ORDER BY early_readmission_rate DESC
        """
    
    @staticmethod
    def get_diagnosis_analysis() -> str:
        """Get diagnosis-based readmission analysis."""
        return """
        SELECT 
            diag_1,
            COUNT(*) as diagnosis_count,
            SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) as early_readmissions,
            SUM(CASE WHEN readmitted = '>30' THEN 1 ELSE 0 END) as late_readmissions,
            SUM(CASE WHEN readmitted = 'NO' THEN 1 ELSE 0 END) as no_readmissions,
            ROUND(
                SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
            ) as early_readmission_rate,
            AVG(time_in_hospital) as avg_length_of_stay
        FROM patients 
        WHERE diag_1 IS NOT NULL AND diag_1 != '?'
        GROUP BY diag_1
        HAVING COUNT(*) >= 10
        ORDER BY early_readmission_rate DESC
        LIMIT 20
        """
    
    @staticmethod
    def get_medication_impact() -> str:
        """Get medication impact on readmission rates."""
        return """
        SELECT 
            m.medication_name,
            COUNT(DISTINCT m.patient_id) as patients_on_medication,
            COUNT(DISTINCT p.patient_id) as total_patients,
            ROUND(
                COUNT(DISTINCT m.patient_id) * 100.0 / COUNT(DISTINCT p.patient_id), 2
            ) as medication_usage_rate,
            SUM(CASE WHEN p.readmitted = '<30' THEN 1 ELSE 0 END) as early_readmissions,
            ROUND(
                SUM(CASE WHEN p.readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / 
                COUNT(DISTINCT m.patient_id), 2
            ) as readmission_rate_with_medication
        FROM medications m
        JOIN patients p ON m.patient_id = p.patient_id
        GROUP BY m.medication_name
        HAVING COUNT(DISTINCT m.patient_id) >= 5
        ORDER BY readmission_rate_with_medication DESC
        """
    
    @staticmethod
    def get_length_of_stay_analysis() -> str:
        """Get length of stay analysis by readmission status."""
        return """
        SELECT 
            CASE 
                WHEN time_in_hospital <= 3 THEN 'Short Stay (1-3 days)'
                WHEN time_in_hospital <= 7 THEN 'Medium Stay (4-7 days)'
                WHEN time_in_hospital <= 14 THEN 'Long Stay (8-14 days)'
                ELSE 'Extended Stay (15+ days)'
            END as stay_category,
            COUNT(*) as patient_count,
            SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) as early_readmissions,
            SUM(CASE WHEN readmitted = '>30' THEN 1 ELSE 0 END) as late_readmissions,
            SUM(CASE WHEN readmitted = 'NO' THEN 1 ELSE 0 END) as no_readmissions,
            ROUND(
                SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
            ) as early_readmission_rate,
            AVG(time_in_hospital) as avg_length_of_stay
        FROM patients 
        GROUP BY 
            CASE 
                WHEN time_in_hospital <= 3 THEN 'Short Stay (1-3 days)'
                WHEN time_in_hospital <= 7 THEN 'Medium Stay (4-7 days)'
                WHEN time_in_hospital <= 14 THEN 'Long Stay (8-14 days)'
                ELSE 'Extended Stay (15+ days)'
            END
        ORDER BY avg_length_of_stay
        """
    
    @staticmethod
    def get_comorbidity_analysis() -> str:
        """Get comorbidity analysis for readmission prediction."""
        return """
        WITH comorbidity_counts AS (
            SELECT 
                patient_id,
                COUNT(CASE WHEN diag_1 IS NOT NULL AND diag_1 != '?' THEN 1 END) +
                COUNT(CASE WHEN diag_2 IS NOT NULL AND diag_2 != '?' THEN 1 END) +
                COUNT(CASE WHEN diag_3 IS NOT NULL AND diag_3 != '?' THEN 1 END) as comorbidity_count
            FROM patients
            GROUP BY patient_id
        )
        SELECT 
            CASE 
                WHEN comorbidity_count = 0 THEN 'No Comorbidities'
                WHEN comorbidity_count = 1 THEN 'Single Comorbidity'
                WHEN comorbidity_count = 2 THEN 'Two Comorbidities'
                ELSE 'Multiple Comorbidities (3+)'
            END as comorbidity_category,
            COUNT(*) as patient_count,
            SUM(CASE WHEN p.readmitted = '<30' THEN 1 ELSE 0 END) as early_readmissions,
            ROUND(
                SUM(CASE WHEN p.readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
            ) as early_readmission_rate,
            AVG(p.time_in_hospital) as avg_length_of_stay,
            AVG(p.num_medications) as avg_medications
        FROM comorbidity_counts cc
        JOIN patients p ON cc.patient_id = p.patient_id
        GROUP BY 
            CASE 
                WHEN comorbidity_count = 0 THEN 'No Comorbidities'
                WHEN comorbidity_count = 1 THEN 'Single Comorbidity'
                WHEN comorbidity_count = 2 THEN 'Two Comorbidities'
                ELSE 'Multiple Comorbidities (3+)'
            END
        ORDER BY comorbidity_count
        """
    
    @staticmethod
    def get_model_performance_trends() -> str:
        """Get model performance trends over time."""
        return """
        SELECT 
            DATE(created_at) as run_date,
            model_name,
            feature_selection_method,
            top_n_features,
            AVG(accuracy) as avg_accuracy,
            AVG(precision) as avg_precision,
            AVG(recall) as avg_recall,
            AVG(f1_score) as avg_f1_score,
            AVG(auc_score) as avg_auc_score,
            COUNT(*) as runs_count
        FROM model_results 
        WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
        GROUP BY DATE(created_at), model_name, feature_selection_method, top_n_features
        ORDER BY run_date DESC, avg_auc_score DESC
        """
    
    @staticmethod
    def get_best_performing_models() -> str:
        """Get best performing models by metric."""
        return """
        WITH ranked_models AS (
            SELECT 
                model_name,
                feature_selection_method,
                top_n_features,
                accuracy,
                precision,
                recall,
                f1_score,
                auc_score,
                created_at,
                ROW_NUMBER() OVER (PARTITION BY model_name ORDER BY auc_score DESC) as rank_by_auc,
                ROW_NUMBER() OVER (PARTITION BY model_name ORDER BY f1_score DESC) as rank_by_f1
            FROM model_results
        )
        SELECT 
            model_name,
            feature_selection_method,
            top_n_features,
            accuracy,
            precision,
            recall,
            f1_score,
            auc_score,
            created_at
        FROM ranked_models
        WHERE rank_by_auc = 1 OR rank_by_f1 = 1
        ORDER BY auc_score DESC, f1_score DESC
        """
    
    @staticmethod
    def get_patient_encounter_history(patient_id: str) -> str:
        """Get patient encounter history for a specific patient."""
        return """
        SELECT 
            p.patient_id,
            p.age,
            p.gender,
            p.race,
            p.admission_type_id,
            p.discharge_disposition_id,
            p.time_in_hospital,
            p.num_medications,
            p.num_lab_procedures,
            p.diag_1,
            p.diag_2,
            p.diag_3,
            p.readmitted,
            p.created_at as encounter_date
        FROM patients p
        WHERE p.patient_id = %(patient_id)s
        ORDER BY p.created_at DESC
        """
    
    @staticmethod
    def get_high_risk_patients(threshold: float = 0.3) -> str:
        """Get patients with high readmission risk based on clinical indicators."""
        return """
        SELECT 
            patient_id,
            age,
            gender,
            time_in_hospital,
            num_medications,
            num_lab_procedures,
            diag_1,
            diag_2,
            diag_3,
            readmitted,
            CASE 
                WHEN num_medications > 10 THEN 1
                ELSE 0
            END + 
            CASE 
                WHEN time_in_hospital > 7 THEN 1
                ELSE 0
            END +
            CASE 
                WHEN num_lab_procedures > 20 THEN 1
                ELSE 0
            END as risk_score
        FROM patients
        WHERE (
            CASE 
                WHEN num_medications > 10 THEN 1
                ELSE 0
            END + 
            CASE 
                WHEN time_in_hospital > 7 THEN 1
                ELSE 0
            END +
            CASE 
                WHEN num_lab_procedures > 20 THEN 1
                ELSE 0
            END
        ) >= %(threshold)s
        ORDER BY risk_score DESC, num_medications DESC
        """
    
    @staticmethod
    def get_operational_metrics() -> str:
        """Get operational metrics for hospital management."""
        return """
        SELECT 
            COUNT(*) as total_patients,
            AVG(time_in_hospital) as avg_length_of_stay,
            SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) as early_readmissions,
            SUM(CASE WHEN readmitted = '>30' THEN 1 ELSE 0 END) as late_readmissions,
            ROUND(
                SUM(CASE WHEN readmitted IN ('<30', '>30') THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
            ) as overall_readmission_rate,
            ROUND(
                SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
            ) as early_readmission_rate,
            AVG(num_medications) as avg_medications_per_patient,
            AVG(num_lab_procedures) as avg_lab_procedures_per_patient,
            COUNT(DISTINCT diag_1) as unique_primary_diagnoses
        FROM patients
        """
    
    @staticmethod
    def get_feature_importance_analysis() -> str:
        """Get analysis of feature importance for model performance."""
        return """
        SELECT 
            feature_selection_method,
            top_n_features,
            COUNT(*) as model_runs,
            AVG(auc_score) as avg_auc_score,
            STDDEV(auc_score) as auc_std_dev,
            AVG(f1_score) as avg_f1_score,
            STDDEV(f1_score) as f1_std_dev,
            MIN(auc_score) as min_auc_score,
            MAX(auc_score) as max_auc_score
        FROM model_results
        GROUP BY feature_selection_method, top_n_features
        ORDER BY avg_auc_score DESC, avg_f1_score DESC
        """


class QueryExecutor:
    """
    Query executor class for running predefined queries with parameters.
    
    This class provides a convenient interface for executing queries
    and handling results in a consistent format.
    """
    
    def __init__(self, db_connector):
        """
        Initialize query executor with database connector.
        
        Args:
            db_connector: DatabaseConnector instance
        """
        self.db_connector = db_connector
        self.queries = HospitalReadmissionQueries()
    
    def execute_demographics_analysis(self) -> pd.DataFrame:
        """Execute patient demographics analysis."""
        query = self.queries.get_patient_demographics()
        return self.db_connector.execute_query(query)
    
    def execute_risk_factors_analysis(self) -> pd.DataFrame:
        """Execute readmission risk factors analysis."""
        query = self.queries.get_readmission_risk_factors()
        return self.db_connector.execute_query(query)
    
    def execute_diagnosis_analysis(self) -> pd.DataFrame:
        """Execute diagnosis-based analysis."""
        query = self.queries.get_diagnosis_analysis()
        return self.db_connector.execute_query(query)
    
    def execute_medication_impact_analysis(self) -> pd.DataFrame:
        """Execute medication impact analysis."""
        query = self.queries.get_medication_impact()
        return self.db_connector.execute_query(query)
    
    def execute_length_of_stay_analysis(self) -> pd.DataFrame:
        """Execute length of stay analysis."""
        query = self.queries.get_length_of_stay_analysis()
        return self.db_connector.execute_query(query)
    
    def execute_comorbidity_analysis(self) -> pd.DataFrame:
        """Execute comorbidity analysis."""
        query = self.queries.get_comorbidity_analysis()
        return self.db_connector.execute_query(query)
    
    def execute_model_performance_trends(self) -> pd.DataFrame:
        """Execute model performance trends analysis."""
        query = self.queries.get_model_performance_trends()
        return self.db_connector.execute_query(query)
    
    def execute_best_performing_models(self) -> pd.DataFrame:
        """Execute best performing models analysis."""
        query = self.queries.get_best_performing_models()
        return self.db_connector.execute_query(query)
    
    def execute_patient_history(self, patient_id: str) -> pd.DataFrame:
        """Execute patient encounter history query."""
        query = self.queries.get_patient_encounter_history(patient_id)
        return self.db_connector.execute_query(query, {'patient_id': patient_id})
    
    def execute_high_risk_patients(self, threshold: float = 0.3) -> pd.DataFrame:
        """Execute high risk patients analysis."""
        query = self.queries.get_high_risk_patients(threshold)
        return self.db_connector.execute_query(query, {'threshold': threshold})
    
    def execute_operational_metrics(self) -> pd.DataFrame:
        """Execute operational metrics query."""
        query = self.queries.get_operational_metrics()
        return self.db_connector.execute_query(query)
    
    def execute_feature_importance_analysis(self) -> pd.DataFrame:
        """Execute feature importance analysis."""
        query = self.queries.get_feature_importance_analysis()
        return self.db_connector.execute_query(query)
    
    def execute_all_analytics(self) -> Dict[str, pd.DataFrame]:
        """
        Execute all analytics queries and return comprehensive results.
        
        Returns:
            Dict: Dictionary containing all analytics DataFrames
        """
        analytics_results = {
            'demographics': self.execute_demographics_analysis(),
            'risk_factors': self.execute_risk_factors_analysis(),
            'diagnosis': self.execute_diagnosis_analysis(),
            'medication_impact': self.execute_medication_impact_analysis(),
            'length_of_stay': self.execute_length_of_stay_analysis(),
            'comorbidity': self.execute_comorbidity_analysis(),
            'model_trends': self.execute_model_performance_trends(),
            'best_models': self.execute_best_performing_models(),
            'operational_metrics': self.execute_operational_metrics(),
            'feature_importance': self.execute_feature_importance_analysis()
        }
        
        return analytics_results


# Example usage
if __name__ == "__main__":
    # This would be used with a database connector
    print("SQL Queries module loaded successfully")
    print("Available query classes:")
    print("- HospitalReadmissionQueries: Predefined queries")
    print("- QueryExecutor: Query execution interface") 