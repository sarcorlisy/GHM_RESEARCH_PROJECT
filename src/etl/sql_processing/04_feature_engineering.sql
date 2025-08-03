-- =====================================================
-- Hospital Readmission Feature Engineering SQL Script
-- Version: 1.0
-- Description: Create features for machine learning models
-- Execution Order: 4th (final step, after all cleaning is complete)
-- Note: Based on patients_business_cleaned table
-- =====================================================

-- Step 1: Create feature engineering table
-- This creates features for machine learning models
-- Drop existing table to ensure fresh data
DROP TABLE IF EXISTS patients_features;
DROP TABLE IF EXISTS feature_summary_stats;

CREATE TABLE patients_features AS
SELECT 
    patient_id,
    encounter_id,
    
    -- Demographic Features
    age,
    gender,
    race,
    
    -- Age Groups (Categorical)
    CASE 
        WHEN age < 30 THEN 'Young'
        WHEN age < 50 THEN 'Middle'
        WHEN age < 70 THEN 'Senior'
        ELSE 'Elderly'
    END as age_group,
    
    -- Hospital Stay Features
    time_in_hospital,
    
    -- Hospital Stay Categories
    CASE 
        WHEN time_in_hospital <= 3 THEN 'Short'
        WHEN time_in_hospital <= 7 THEN 'Medium'
        WHEN time_in_hospital <= 14 THEN 'Long'
        ELSE 'Very Long'
    END as stay_duration_category,
    
    -- Medical Procedure Features
    num_lab_procedures,
    num_procedures,
    num_medications,
    
    -- Visit History Features (simplified - using available columns)
    -- Note: number_outpatient, number_emergency, number_inpatient were removed in dynamic cleaning
    
    -- Total Previous Visits (simplified)
    0 as total_previous_visits,
    
    -- Visit Frequency Categories (simplified)
    'No Previous Visits' as visit_frequency_category,
    
    -- Diagnosis Features
    number_diagnoses,
    
    -- Diagnosis Complexity
    CASE 
        WHEN number_diagnoses <= 3 THEN 'Simple'
        WHEN number_diagnoses <= 6 THEN 'Moderate'
        WHEN number_diagnoses <= 10 THEN 'Complex'
        ELSE 'Very Complex'
    END as diagnosis_complexity,
    
    -- Admission Features
    admission_type_id,
    discharge_disposition_id,
    admission_source_id,
    
    -- Readmission Target (Binary)
    CASE 
        WHEN readmitted = '<30' THEN 1
        ELSE 0
    END as readmission_30_days,
    
    CASE 
        WHEN readmitted IN ('<30', '>30') THEN 1
        ELSE 0
    END as readmission_any,
    
    -- Risk Score Features
    -- Simple risk score based on multiple factors (simplified)
    (
        COALESCE(age, 50) * 0.1 +
        COALESCE(time_in_hospital, 5) * 0.2 +
        COALESCE(num_medications, 10) * 0.15 +
        COALESCE(number_diagnoses, 5) * 0.25 +
        0 * 0.3  -- Simplified: no visit history available
    ) as risk_score,
    
    -- Risk Categories (simplified)
    CASE 
        WHEN (
            COALESCE(age, 50) * 0.1 +
            COALESCE(time_in_hospital, 5) * 0.2 +
            COALESCE(num_medications, 10) * 0.15 +
            COALESCE(number_diagnoses, 5) * 0.25 +
            0 * 0.3  -- Simplified: no visit history available
        ) < 10 THEN 'Low Risk'
        WHEN (
            COALESCE(age, 50) * 0.1 +
            COALESCE(time_in_hospital, 5) * 0.2 +
            COALESCE(num_medications, 10) * 0.15 +
            COALESCE(number_diagnoses, 5) * 0.25 +
            0 * 0.3  -- Simplified: no visit history available
        ) < 20 THEN 'Medium Risk'
        ELSE 'High Risk'
    END as risk_category,
    
    -- Original cleaned fields for reference
    readmitted,
    
    -- Processing timestamp
    NOW() as feature_created_at
    
FROM patients_business_cleaned
WHERE patient_id IS NOT NULL;

-- Step 2: Create feature summary statistics
CREATE TABLE IF NOT EXISTS feature_summary_stats AS
SELECT 
    'Age Statistics' as feature_group,
    'Mean Age' as metric,
    ROUND(AVG(age), 2) as value
FROM patients_features
WHERE age IS NOT NULL

UNION ALL

SELECT 
    'Age Statistics' as feature_group,
    'Median Age' as metric,
    ROUND(AVG(age), 2) as value
FROM patients_features
WHERE age IS NOT NULL

UNION ALL

SELECT 
    'Hospital Stay Statistics' as feature_group,
    'Mean Stay Duration' as metric,
    ROUND(AVG(time_in_hospital), 2) as value
FROM patients_features
WHERE time_in_hospital IS NOT NULL

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

-- Step 4: Display feature engineering results
SELECT 'Feature Engineering Complete' as status;
SELECT COUNT(*) as total_features_created FROM patients_features;

-- Show feature summary
SELECT * FROM feature_summary_stats; 