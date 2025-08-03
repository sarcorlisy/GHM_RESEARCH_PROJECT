-- =====================================================
-- Hospital Readmission Business Rules Cleaning SQL Script
-- Version: 1.0
-- Description: Apply business rules and final data cleaning
-- Execution Order: 3rd (after dynamic column cleaning in Python)
-- Note: Dynamic column cleaning already completed in Python
-- =====================================================

-- Step 1: Create business rules cleaned table
-- This applies business rules to remove patients who cannot be readmitted
-- Note: Dynamic column cleaning (removing high invalid rate columns) already done in Python
-- Drop existing table to ensure fresh data
DROP TABLE IF EXISTS patients_business_cleaned;
DROP TABLE IF EXISTS business_rules_summary;
DROP TABLE IF EXISTS removed_records_summary;
DROP TABLE IF EXISTS final_data_quality_metrics;

CREATE TABLE patients_business_cleaned AS
SELECT *
FROM patients_cleaned
WHERE discharge_disposition_desc NOT IN (
    'Expired',
    'Hospice / home', 
    'Hospice / medical facility',
    'Expired at home. Medicaid only, hospice.',
    'Expired in a medical facility. Medicaid only, hospice.',
    'Expired, place unknown. Medicaid only, hospice.'
);

-- Step 2: Create indexes for better performance
CREATE INDEX idx_patient_nbr_business_cleaned ON patients_business_cleaned(patient_nbr);
CREATE INDEX idx_readmitted_business_cleaned ON patients_business_cleaned(readmitted);
CREATE INDEX idx_age_business_cleaned ON patients_business_cleaned(age);
CREATE INDEX idx_time_in_hospital_business_cleaned ON patients_business_cleaned(time_in_hospital);

-- Step 3: Generate business rules cleaning summary
CREATE TABLE IF NOT EXISTS business_rules_summary AS
SELECT 
    'Original Records (after mapping)' as stage,
    COUNT(*) as record_count
FROM patients_mapped

UNION ALL

SELECT 
    'After Basic Cleaning' as stage,
    COUNT(*) as record_count
FROM patients_cleaned

UNION ALL

SELECT 
    'After Business Rules' as stage,
    COUNT(*) as record_count
FROM patients_business_cleaned;

-- Step 4: Show removed records statistics
CREATE TABLE IF NOT EXISTS removed_records_summary AS
SELECT 
    'Removed Records (Business Rules)' as category,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM patients_cleaned), 2) as percentage
FROM patients_cleaned
WHERE discharge_disposition_desc IN (
    'Expired',
    'Hospice / home', 
    'Hospice / medical facility',
    'Expired at home. Medicaid only, hospice.',
    'Expired in a medical facility. Medicaid only, hospice.',
    'Expired, place unknown. Medicaid only, hospice.'
);

-- Step 5: Show final data quality metrics
CREATE TABLE IF NOT EXISTS final_data_quality_metrics AS
SELECT 
    'Total Records (Final)' as metric,
    COUNT(*) as value
FROM patients_business_cleaned

UNION ALL

SELECT 
    'Records with Valid Age' as metric,
    COUNT(*) as value
FROM patients_business_cleaned
WHERE age IS NOT NULL AND age >= 0 AND age <= 120

UNION ALL

SELECT 
    'Records with Valid Gender' as metric,
    COUNT(*) as value
FROM patients_business_cleaned
WHERE gender IS NOT NULL AND gender != 'Unknown/Invalid'

UNION ALL

SELECT 
    'Records with Valid Readmission Status' as metric,
    COUNT(*) as value
FROM patients_business_cleaned
WHERE readmitted IS NOT NULL AND readmitted != '?'

UNION ALL

SELECT 
    'Records with Valid Time in Hospital' as metric,
    COUNT(*) as value
FROM patients_business_cleaned
WHERE time_in_hospital IS NOT NULL AND time_in_hospital >= 0 AND time_in_hospital <= 365;

-- Step 6: Display summary results
SELECT 'Business Rules Cleaning Complete' as status;
SELECT COUNT(*) as final_record_count FROM patients_business_cleaned;

-- Show the summary tables
SELECT * FROM business_rules_summary;
SELECT * FROM removed_records_summary;
SELECT * FROM final_data_quality_metrics; 