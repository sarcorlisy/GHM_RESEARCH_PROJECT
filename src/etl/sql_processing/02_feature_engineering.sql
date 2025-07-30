-- =====================================================
-- Hospital Readmission Feature Engineering SQL Script
-- Version: 1.0
-- Description: Create features for machine learning models
-- =====================================================

-- Step 1: Create feature engineering table
CREATE TABLE IF NOT EXISTS patients_features AS
SELECT 
    patient_id,
    encounter_id,
    
    -- Demographic Features
    age_cleaned,
    gender_cleaned,
    race_cleaned,
    
    -- Age Groups (Categorical)
    CASE 
        WHEN age_cleaned < 30 THEN 'Young'
        WHEN age_cleaned < 50 THEN 'Middle'
        WHEN age_cleaned < 70 THEN 'Senior'
        ELSE 'Elderly'
    END as age_group,
    
    -- Hospital Stay Features
    time_in_hospital_cleaned,
    
    -- Hospital Stay Categories
    CASE 
        WHEN time_in_hospital_cleaned <= 3 THEN 'Short'
        WHEN time_in_hospital_cleaned <= 7 THEN 'Medium'
        WHEN time_in_hospital_cleaned <= 14 THEN 'Long'
        ELSE 'Very Long'
    END as stay_duration_category,
    
    -- Medical Procedure Features
    num_lab_procedures_cleaned,
    num_procedures_cleaned,
    num_medications_cleaned,
    
    -- Visit History Features
    number_outpatient_cleaned,
    number_emergency_cleaned,
    number_inpatient_cleaned,
    
    -- Total Previous Visits
    COALESCE(number_outpatient_cleaned, 0) + 
    COALESCE(number_emergency_cleaned, 0) + 
    COALESCE(number_inpatient_cleaned, 0) as total_previous_visits,
    
    -- Visit Frequency Categories
    CASE 
        WHEN (COALESCE(number_outpatient_cleaned, 0) + 
              COALESCE(number_emergency_cleaned, 0) + 
              COALESCE(number_inpatient_cleaned, 0)) = 0 THEN 'No Previous Visits'
        WHEN (COALESCE(number_outpatient_cleaned, 0) + 
              COALESCE(number_emergency_cleaned, 0) + 
              COALESCE(number_inpatient_cleaned, 0)) <= 2 THEN 'Low Frequency'
        WHEN (COALESCE(number_outpatient_cleaned, 0) + 
              COALESCE(number_emergency_cleaned, 0) + 
              COALESCE(number_inpatient_cleaned, 0)) <= 5 THEN 'Medium Frequency'
        ELSE 'High Frequency'
    END as visit_frequency_category,
    
    -- Diagnosis Features
    number_diagnoses_cleaned,
    
    -- Diagnosis Complexity
    CASE 
        WHEN number_diagnoses_cleaned <= 3 THEN 'Simple'
        WHEN number_diagnoses_cleaned <= 6 THEN 'Moderate'
        WHEN number_diagnoses_cleaned <= 10 THEN 'Complex'
        ELSE 'Very Complex'
    END as diagnosis_complexity,
    
    -- Admission Features
    admission_type_id_cleaned,
    discharge_disposition_id_cleaned,
    admission_source_id_cleaned,
    
    -- Readmission Target (Binary)
    CASE 
        WHEN readmitted_cleaned = '<30' THEN 1
        ELSE 0
    END as readmission_30_days,
    
    CASE 
        WHEN readmitted_cleaned IN ('<30', '>30') THEN 1
        ELSE 0
    END as readmission_any,
    
    -- Risk Score Features
    -- Simple risk score based on multiple factors
    (
        COALESCE(age_cleaned, 50) * 0.1 +
        COALESCE(time_in_hospital_cleaned, 5) * 0.2 +
        COALESCE(num_medications_cleaned, 10) * 0.15 +
        COALESCE(number_diagnoses_cleaned, 5) * 0.25 +
        (COALESCE(number_outpatient_cleaned, 0) + 
         COALESCE(number_emergency_cleaned, 0) + 
         COALESCE(number_inpatient_cleaned, 0)) * 0.3
    ) as risk_score,
    
    -- Risk Categories
    CASE 
        WHEN (
            COALESCE(age_cleaned, 50) * 0.1 +
            COALESCE(time_in_hospital_cleaned, 5) * 0.2 +
            COALESCE(num_medications_cleaned, 10) * 0.15 +
            COALESCE(number_diagnoses_cleaned, 5) * 0.25 +
            (COALESCE(number_outpatient_cleaned, 0) + 
             COALESCE(number_emergency_cleaned, 0) + 
             COALESCE(number_inpatient_cleaned, 0)) * 0.3
        ) < 10 THEN 'Low Risk'
        WHEN (
            COALESCE(age_cleaned, 50) * 0.1 +
            COALESCE(time_in_hospital_cleaned, 5) * 0.2 +
            COALESCE(num_medications_cleaned, 10) * 0.15 +
            COALESCE(number_diagnoses_cleaned, 5) * 0.25 +
            (COALESCE(number_outpatient_cleaned, 0) + 
             COALESCE(number_emergency_cleaned, 0) + 
             COALESCE(number_inpatient_cleaned, 0)) * 0.3
        ) < 20 THEN 'Medium Risk'
        ELSE 'High Risk'
    END as risk_category,
    
    -- Original cleaned fields for reference
    readmitted_cleaned,
    
    -- Processing timestamp
    NOW() as feature_created_at
    
FROM patients_cleaned
WHERE patient_id IS NOT NULL;

-- Step 2: Create feature summary statistics
CREATE TABLE IF NOT EXISTS feature_summary_stats AS
SELECT 
    'Age Statistics' as feature_group,
    'Mean Age' as metric,
    ROUND(AVG(age_cleaned), 2) as value
FROM patients_features
WHERE age_cleaned IS NOT NULL

UNION ALL

SELECT 
    'Age Statistics' as feature_group,
    'Median Age' as metric,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY age_cleaned), 2) as value
FROM patients_features
WHERE age_cleaned IS NOT NULL

UNION ALL

SELECT 
    'Hospital Stay Statistics' as feature_group,
    'Mean Stay Duration' as metric,
    ROUND(AVG(time_in_hospital_cleaned), 2) as value
FROM patients_features
WHERE time_in_hospital_cleaned IS NOT NULL

UNION ALL

SELECT 
    'Readmission Statistics' as feature_group,
    '30-Day Readmission Rate' as metric,
    ROUND(AVG(readmission_30_days) * 100, 2) as value
FROM patients_features

UNION ALL

SELECT 
    'Readmission Statistics' as feature_group,
    'Any Readmission Rate' as metric,
    ROUND(AVG(readmission_any) * 100, 2) as value
FROM patients_features

UNION ALL

SELECT 
    'Risk Score Statistics' as feature_group,
    'Mean Risk Score' as metric,
    ROUND(AVG(risk_score), 2) as value
FROM patients_features;

-- Step 3: Create indexes for feature table
CREATE INDEX idx_patient_id_features ON patients_features(patient_id);
CREATE INDEX idx_readmission_30_days ON patients_features(readmission_30_days);
CREATE INDEX idx_readmission_any ON patients_features(readmission_any);
CREATE INDEX idx_risk_category ON patients_features(risk_category);
CREATE INDEX idx_age_group ON patients_features(age_group);
CREATE INDEX idx_stay_duration_category ON patients_features(stay_duration_category); 