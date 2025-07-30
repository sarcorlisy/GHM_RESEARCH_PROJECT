-- =====================================================
-- Hospital Readmission Data Cleaning SQL Script
-- Version: 1.0
-- Description: Clean and prepare raw patient data
-- =====================================================

-- Step 1: Create cleaned patients table with data quality improvements
CREATE TABLE IF NOT EXISTS patients_cleaned AS
SELECT 
    patient_id,
    id as encounter_id,
    -- Clean age field (handle age ranges and invalid values)
    CASE 
        WHEN age IS NULL THEN NULL
        WHEN age < 0 THEN NULL
        WHEN age > 120 THEN NULL
        ELSE age 
    END as age_cleaned,
    
    -- Clean gender field
    CASE 
        WHEN gender = 'Unknown/Invalid' THEN 'Unknown'
        WHEN gender IS NULL THEN 'Unknown'
        ELSE gender 
    END as gender_cleaned,
    
    -- Clean race field
    CASE 
        WHEN race = '?' THEN 'Unknown'
        WHEN race IS NULL THEN 'Unknown'
        ELSE race 
    END as race_cleaned,
    
    -- Clean admission type
    CASE 
        WHEN admission_type_id = '?' THEN NULL
        ELSE CAST(admission_type_id AS UNSIGNED)
    END as admission_type_id_cleaned,
    
    -- Clean discharge disposition
    CASE 
        WHEN discharge_disposition_id = '?' THEN NULL
        ELSE CAST(discharge_disposition_id AS UNSIGNED)
    END as discharge_disposition_id_cleaned,
    
    -- Clean admission source
    CASE 
        WHEN admission_source_id = '?' THEN NULL
        ELSE CAST(admission_source_id AS UNSIGNED)
    END as admission_source_id_cleaned,
    
    -- Clean time in hospital
    CASE 
        WHEN time_in_hospital < 0 THEN NULL
        WHEN time_in_hospital > 365 THEN NULL
        ELSE time_in_hospital 
    END as time_in_hospital_cleaned,
    
    -- Clean number of lab procedures
    CASE 
        WHEN num_lab_procedures < 0 THEN NULL
        ELSE num_lab_procedures 
    END as num_lab_procedures_cleaned,
    
    -- Clean number of procedures
    CASE 
        WHEN num_procedures < 0 THEN NULL
        ELSE num_procedures 
    END as num_procedures_cleaned,
    
    -- Clean number of medications
    CASE 
        WHEN num_medications < 0 THEN NULL
        ELSE num_medications 
    END as num_medications_cleaned,
    
    -- Clean number of outpatient visits
    CASE 
        WHEN number_outpatient < 0 THEN NULL
        ELSE number_outpatient 
    END as number_outpatient_cleaned,
    
    -- Clean number of emergency visits
    CASE 
        WHEN number_emergency < 0 THEN NULL
        ELSE number_emergency 
    END as number_emergency_cleaned,
    
    -- Clean number of inpatient visits
    CASE 
        WHEN number_inpatient < 0 THEN NULL
        ELSE number_inpatient 
    END as number_inpatient_cleaned,
    
    -- Count diagnoses (diag_1, diag_2, diag_3)
    CASE 
        WHEN diag_1 IS NOT NULL AND diag_1 != '?' THEN 1
        ELSE 0
    END +
    CASE 
        WHEN diag_2 IS NOT NULL AND diag_2 != '?' THEN 1
        ELSE 0
    END +
    CASE 
        WHEN diag_3 IS NOT NULL AND diag_3 != '?' THEN 1
        ELSE 0
    END as number_diagnoses_cleaned,
    
    -- Clean readmitted field
    CASE 
        WHEN readmitted = '?' THEN 'Unknown'
        WHEN readmitted IS NULL THEN 'Unknown'
        ELSE readmitted 
    END as readmitted_cleaned,
    
    -- Keep original fields for reference
    age,
    gender,
    race,
    admission_type_id,
    discharge_disposition_id,
    admission_source_id,
    time_in_hospital,
    num_lab_procedures,
    num_procedures,
    num_medications,
    number_outpatient,
    number_emergency,
    number_inpatient,
    number_diagnoses,
    readmitted,
    
    -- Add data quality flags
    CASE 
        WHEN age IS NULL OR age < 0 OR age > 120 THEN 1
        ELSE 0 
    END as age_quality_flag,
    
    CASE 
        WHEN gender = 'Unknown/Invalid' OR gender IS NULL THEN 1
        ELSE 0 
    END as gender_quality_flag,
    
    -- Add processing timestamp
    NOW() as processed_at
    
FROM patients
WHERE patient_id IS NOT NULL;  -- Remove records without patient ID

-- Step 2: Create data quality summary
CREATE TABLE IF NOT EXISTS data_quality_summary AS
SELECT 
    'Total Records' as metric,
    COUNT(*) as value
FROM patients_cleaned

UNION ALL

SELECT 
    'Records with Clean Age' as metric,
    COUNT(*) as value
FROM patients_cleaned
WHERE age_quality_flag = 0

UNION ALL

SELECT 
    'Records with Clean Gender' as metric,
    COUNT(*) as value
FROM patients_cleaned
WHERE gender_quality_flag = 0

UNION ALL

SELECT 
    'Records with Missing Age' as metric,
    COUNT(*) as value
FROM patients_cleaned
WHERE age_quality_flag = 1

UNION ALL

SELECT 
    'Records with Missing Gender' as metric,
    COUNT(*) as value
FROM patients_cleaned
WHERE gender_quality_flag = 1;

-- Step 3: Create indexes for better performance
CREATE INDEX idx_patient_id_cleaned ON patients_cleaned(patient_id);
CREATE INDEX idx_age_cleaned ON patients_cleaned(age_cleaned);
CREATE INDEX idx_gender_cleaned ON patients_cleaned(gender_cleaned);
CREATE INDEX idx_readmitted_cleaned ON patients_cleaned(readmitted_cleaned);
CREATE INDEX idx_time_in_hospital_cleaned ON patients_cleaned(time_in_hospital_cleaned); 