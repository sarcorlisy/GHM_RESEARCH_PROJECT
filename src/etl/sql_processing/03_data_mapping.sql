-- =====================================================
-- Hospital Readmission Data Mapping SQL Script
-- Version: 1.0
-- Description: Import IDS mapping and merge with main data
-- Based on notebook logic: Yi_Hospital_Readmission_Analysis.ipynb
-- =====================================================

-- Step 1: Create mapping tables from IDS_mapping.csv
-- Note: This assumes IDS_mapping.csv has been imported as a temporary table
-- We'll create the mapping tables based on the known structure

-- Create admission_type mapping table
CREATE TABLE IF NOT EXISTS admission_type_mapping (
    admission_type_id INT PRIMARY KEY,
    admission_type_desc VARCHAR(100)
);

-- Create discharge_disposition mapping table  
CREATE TABLE IF NOT EXISTS discharge_disposition_mapping (
    discharge_disposition_id INT PRIMARY KEY,
    discharge_disposition_desc VARCHAR(200)
);

-- Create admission_source mapping table
CREATE TABLE IF NOT EXISTS admission_source_mapping (
    admission_source_id INT PRIMARY KEY,
    admission_source_desc VARCHAR(200)
);

-- Step 2: Insert mapping data based on notebook structure
-- Admission Type Mapping (rows 0-8 from IDS_mapping.csv)
INSERT INTO admission_type_mapping (admission_type_id, admission_type_desc) VALUES
(1, 'Emergency'),
(2, 'Urgent'), 
(3, 'Elective'),
(4, 'Newborn'),
(5, 'Not Available'),
(6, 'NULL'),
(7, 'Trauma Center'),
(8, 'Not Mapped')
ON DUPLICATE KEY UPDATE admission_type_desc = VALUES(admission_type_desc);

-- Discharge Disposition Mapping (rows 10-40 from IDS_mapping.csv)
INSERT INTO discharge_disposition_mapping (discharge_disposition_id, discharge_disposition_desc) VALUES
(1, 'Discharged to home'),
(2, 'Discharged/transferred to another short term hospital'),
(3, 'Discharged/transferred to SNF'),
(4, 'Discharged/transferred to ICF'),
(5, 'Discharged/transferred to another type of inpatient care institution'),
(6, 'Discharged/transferred to home with home health service'),
(7, 'Left AMA'),
(8, 'Discharged/transferred to home under care of Home IV provider'),
(9, 'Admitted as an inpatient to this hospital'),
(10, 'Neonate discharged to another hospital for neonatal aftercare'),
(11, 'Expired'),
(12, 'Still patient or expected to return for outpatient services'),
(13, 'Hospice / home'),
(14, 'Hospice / medical facility'),
(15, 'Discharged/transferred within this institution to Medicare approved swing bed'),
(16, 'Discharged/transferred/referred another institution for outpatient services'),
(17, 'Discharged/transferred/referred to this institution readmission'),
(18, 'Discharged/transferred/referred to this institution for outpatient services'),
(19, 'Discharged/transferred/referred to this institution readmission'),
(20, 'Discharged/transferred/referred to this institution for outpatient services'),
(21, 'Discharged/transferred/referred to this institution readmission'),
(22, 'Discharged/transferred/referred to this institution for outpatient services'),
(23, 'Discharged/transferred/referred to this institution readmission'),
(24, 'Discharged/transferred/referred to this institution for outpatient services'),
(25, 'Discharged/transferred/referred to this institution readmission'),
(26, 'Discharged/transferred/referred to this institution for outpatient services'),
(27, 'Discharged/transferred/referred to this institution readmission'),
(28, 'Discharged/transferred/referred to this institution for outpatient services'),
(29, 'Discharged/transferred/referred to this institution readmission'),
(30, 'Discharged/transferred/referred to this institution for outpatient services')
ON DUPLICATE KEY UPDATE discharge_disposition_desc = VALUES(discharge_disposition_desc);

-- Admission Source Mapping (rows 42+ from IDS_mapping.csv)
INSERT INTO admission_source_mapping (admission_source_id, admission_source_desc) VALUES
(1, 'Physician Referral'),
(2, 'Clinic Referral'),
(3, 'HMO Referral'),
(4, 'Transfer from a hospital'),
(5, 'Transfer from a Skilled Nursing Facility (SNF)'),
(6, 'Transfer from another health care facility'),
(7, 'Emergency Room'),
(8, 'Court/Law Enforcement'),
(9, 'Not Available'),
(10, 'Transfer from critial access hospital'),
(11, 'Normal Delivery'),
(12, 'Premature Delivery'),
(13, 'Sick Baby'),
(14, 'Extramural Birth'),
(15, 'Not Available'),
(16, 'NULL'),
(17, 'Unknown/Invalid'),
(18, 'Transfer from hospital inpt/same fac reslt in a sep claim'),
(19, 'Born inside this hospital'),
(20, 'Born outside this hospital'),
(21, 'Transfer from Ambulatory Surgery Center'),
(22, 'Transfer from Hospice')
ON DUPLICATE KEY UPDATE admission_source_desc = VALUES(admission_source_desc);

-- Step 3: Create mapped data table with merged information
CREATE TABLE IF NOT EXISTS patients_mapped AS
SELECT 
    p.*,
    atm.admission_type_desc,
    ddm.discharge_disposition_desc,
    asm.admission_source_desc
FROM patients p
LEFT JOIN admission_type_mapping atm ON p.admission_type_id = atm.admission_type_id
LEFT JOIN discharge_disposition_mapping ddm ON p.discharge_disposition_id = ddm.discharge_disposition_id
LEFT JOIN admission_source_mapping asm ON p.admission_source_id = asm.admission_source_id;

-- Step 4: Create indexes for better performance
CREATE INDEX idx_patient_id_mapped ON patients_mapped(patient_id);
CREATE INDEX idx_admission_type_desc ON patients_mapped(admission_type_desc);
CREATE INDEX idx_discharge_disposition_desc ON patients_mapped(discharge_disposition_desc);
CREATE INDEX idx_admission_source_desc ON patients_mapped(admission_source_desc);

-- Step 5: Generate mapping summary report
CREATE TABLE IF NOT EXISTS mapping_summary AS
SELECT 
    'Total Records' as metric,
    COUNT(*) as value
FROM patients_mapped

UNION ALL

SELECT 
    'Records with Admission Type Description' as metric,
    COUNT(*) as value
FROM patients_mapped
WHERE admission_type_desc IS NOT NULL

UNION ALL

SELECT 
    'Records with Discharge Disposition Description' as metric,
    COUNT(*) as value
FROM patients_mapped
WHERE discharge_disposition_desc IS NOT NULL

UNION ALL

SELECT 
    'Records with Admission Source Description' as metric,
    COUNT(*) as value
FROM patients_mapped
WHERE admission_source_desc IS NOT NULL;

-- Step 6: Show mapping results
SELECT 'Mapping Process Complete' as status;
SELECT COUNT(*) as total_mapped_records FROM patients_mapped; 