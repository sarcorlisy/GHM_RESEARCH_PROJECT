-- Hospital Readmission Data Cleaning SQL Script
-- 基于notebook逻辑的数据清洗

-- 步骤1: 删除被mapping的ID列（避免冗余）
-- 创建临时表，不包含ID列
CREATE TABLE IF NOT EXISTS patients_cleaned AS
SELECT 
    id,
    encounter_id,
    patient_id,
    race,
    gender,
    age,
    weight,
    -- 删除 admission_type_id, discharge_disposition_id, admission_source_id
    time_in_hospital,
    payer_code,
    medical_specialty,
    num_lab_procedures,
    num_procedures,
    num_medications,
    number_outpatient,
    number_emergency,
    number_inpatient,
    diag_1,
    diag_2,
    diag_3,
    number_diagnoses,
    max_glu_serum,
    A1Cresult,
    metformin,
    repaglinide,
    nateglinide,
    chlorpropamide,
    glimepiride,
    acetohexamide,
    glipizide,
    glyburide,
    tolbutamide,
    pioglitazone,
    rosiglitazone,
    acarbose,
    miglitol,
    troglitazone,
    tolazamide,
    examide,
    citoglipton,
    insulin,
    `glyburide-metformin`,
    `glipizide-metformin`,
    `glimepiride-pioglitazone`,
    `metformin-rosiglitazone`,
    `metformin-pioglitazone`,
    medication_change,
    diabetesMed,
    readmitted,
    created_at,
    -- 保留映射描述列
    admission_type_desc,
    discharge_disposition_desc,
    admission_source_desc
FROM patients_mapped;

-- 步骤2: 分析无效值比例（包括'Unknown'、'Not Available'等无效值）
-- 创建无效值分析表
CREATE TABLE IF NOT EXISTS invalid_value_analysis AS
SELECT 
    'weight' as column_name,
    COUNT(CASE WHEN weight IN ('Unknown', 'Not Available', 'NULL', '') THEN 1 END) as invalid_count,
    COUNT(*) as total_count,
    ROUND(COUNT(CASE WHEN weight IN ('Unknown', 'Not Available', 'NULL', '') THEN 1 END) * 100.0 / COUNT(*), 2) as invalid_percentage
FROM patients_cleaned
UNION ALL
SELECT 
    'payer_code' as column_name,
    COUNT(CASE WHEN payer_code IN ('Unknown', 'Not Available', 'NULL', '') THEN 1 END) as invalid_count,
    COUNT(*) as total_count,
    ROUND(COUNT(CASE WHEN payer_code IN ('Unknown', 'Not Available', 'NULL', '') THEN 1 END) * 100.0 / COUNT(*), 2) as invalid_percentage
FROM patients_cleaned
UNION ALL
SELECT 
    'medical_specialty' as column_name,
    COUNT(CASE WHEN medical_specialty IN ('Unknown', 'Not Available', 'NULL', '') THEN 1 END) as invalid_count,
    COUNT(*) as total_count,
    ROUND(COUNT(CASE WHEN medical_specialty IN ('Unknown', 'Not Available', 'NULL', '') THEN 1 END) * 100.0 / COUNT(*), 2) as invalid_percentage
FROM patients_cleaned
UNION ALL
SELECT 
    'race' as column_name,
    COUNT(CASE WHEN race IN ('Unknown', 'Not Available', 'NULL', '') THEN 1 END) as invalid_count,
    COUNT(*) as total_count,
    ROUND(COUNT(CASE WHEN race IN ('Unknown', 'Not Available', 'NULL', '') THEN 1 END) * 100.0 / COUNT(*), 2) as invalid_percentage
FROM patients_cleaned
UNION ALL
SELECT 
    'diag_1' as column_name,
    COUNT(CASE WHEN diag_1 IN ('Unknown', 'Not Available', 'NULL', '') THEN 1 END) as invalid_count,
    COUNT(*) as total_count,
    ROUND(COUNT(CASE WHEN diag_1 IN ('Unknown', 'Not Available', 'NULL', '') THEN 1 END) * 100.0 / COUNT(*), 2) as invalid_percentage
FROM patients_cleaned
UNION ALL
SELECT 
    'diag_2' as column_name,
    COUNT(CASE WHEN diag_2 IN ('Unknown', 'Not Available', 'NULL', '') THEN 1 END) as invalid_count,
    COUNT(*) as total_count,
    ROUND(COUNT(CASE WHEN diag_2 IN ('Unknown', 'Not Available', 'NULL', '') THEN 1 END) * 100.0 / COUNT(*), 2) as invalid_percentage
FROM patients_cleaned
UNION ALL
SELECT 
    'diag_3' as column_name,
    COUNT(CASE WHEN diag_3 IN ('Unknown', 'Not Available', 'NULL', '') THEN 1 END) as invalid_count,
    COUNT(*) as total_count,
    ROUND(COUNT(CASE WHEN diag_3 IN ('Unknown', 'Not Available', 'NULL', '') THEN 1 END) * 100.0 / COUNT(*), 2) as invalid_percentage
FROM patients_cleaned
UNION ALL
SELECT 
    'max_glu_serum' as column_name,
    COUNT(CASE WHEN max_glu_serum IN ('Unknown', 'Not Available', 'NULL', '') THEN 1 END) as invalid_count,
    COUNT(*) as total_count,
    ROUND(COUNT(CASE WHEN max_glu_serum IN ('Unknown', 'Not Available', 'NULL', '') THEN 1 END) * 100.0 / COUNT(*), 2) as invalid_percentage
FROM patients_cleaned
UNION ALL
SELECT 
    'A1Cresult' as column_name,
    COUNT(CASE WHEN A1Cresult IN ('Unknown', 'Not Available', 'NULL', '') THEN 1 END) as invalid_count,
    COUNT(*) as total_count,
    ROUND(COUNT(CASE WHEN A1Cresult IN ('Unknown', 'Not Available', 'NULL', '') THEN 1 END) * 100.0 / COUNT(*), 2) as invalid_percentage
FROM patients_cleaned;

-- 显示无效值分析结果
SELECT * FROM invalid_value_analysis ORDER BY invalid_percentage DESC;

-- 显示无效率超过50%的列
SELECT 
    'High Invalid Rate Columns (>50%)' as analysis_type,
    GROUP_CONCAT(column_name SEPARATOR ', ') as columns_to_remove,
    COUNT(*) as column_count
FROM invalid_value_analysis 
WHERE invalid_percentage > 50;

-- 步骤3: 根据无效值分析删除高无效率的列，并处理剩余列
-- 创建最终清洗表（排除无效率超过50%的列）
CREATE TABLE IF NOT EXISTS patients_final_cleaned AS
SELECT 
    id,
    encounter_id,
    patient_id,
    race,
    gender,
    age,
    -- 根据无效值分析决定是否保留weight列
    CASE 
        WHEN (SELECT invalid_percentage FROM invalid_value_analysis WHERE column_name = 'weight') <= 50 
        THEN weight 
        ELSE NULL 
    END as weight,
    time_in_hospital,
    -- 根据无效值分析决定是否保留payer_code列
    CASE 
        WHEN (SELECT invalid_percentage FROM invalid_value_analysis WHERE column_name = 'payer_code') <= 50 
        THEN payer_code 
        ELSE NULL 
    END as payer_code,
    -- 根据无效值分析决定是否保留medical_specialty列
    CASE 
        WHEN (SELECT invalid_percentage FROM invalid_value_analysis WHERE column_name = 'medical_specialty') <= 50 
        THEN medical_specialty 
        ELSE NULL 
    END as medical_specialty,
    num_lab_procedures,
    num_procedures,
    num_medications,
    number_outpatient,
    number_emergency,
    number_inpatient,
    -- 根据无效值分析决定是否保留诊断列
    CASE 
        WHEN (SELECT invalid_percentage FROM invalid_value_analysis WHERE column_name = 'diag_1') <= 50 
        THEN diag_1 
        ELSE NULL 
    END as diag_1,
    CASE 
        WHEN (SELECT invalid_percentage FROM invalid_value_analysis WHERE column_name = 'diag_2') <= 50 
        THEN diag_2 
        ELSE NULL 
    END as diag_2,
    CASE 
        WHEN (SELECT invalid_percentage FROM invalid_value_analysis WHERE column_name = 'diag_3') <= 50 
        THEN diag_3 
        ELSE NULL 
    END as diag_3,
    number_diagnoses,
    -- 根据无效值分析决定是否保留max_glu_serum列
    CASE 
        WHEN (SELECT invalid_percentage FROM invalid_value_analysis WHERE column_name = 'max_glu_serum') <= 50 
        THEN max_glu_serum 
        ELSE NULL 
    END as max_glu_serum,
    -- 根据无效值分析决定是否保留A1Cresult列
    CASE 
        WHEN (SELECT invalid_percentage FROM invalid_value_analysis WHERE column_name = 'A1Cresult') <= 50 
        THEN A1Cresult 
        ELSE NULL 
    END as A1Cresult,
    metformin,
    repaglinide,
    nateglinide,
    chlorpropamide,
    glimepiride,
    acetohexamide,
    glipizide,
    glyburide,
    tolbutamide,
    pioglitazone,
    rosiglitazone,
    acarbose,
    miglitol,
    troglitazone,
    tolazamide,
    examide,
    citoglipton,
    insulin,
    `glyburide-metformin`,
    `glipizide-metformin`,
    `glimepiride-pioglitazone`,
    `metformin-rosiglitazone`,
    `metformin-pioglitazone`,
    medication_change,
    diabetesMed,
    readmitted,
    created_at,
    admission_type_desc,
    discharge_disposition_desc,
    admission_source_desc
FROM patients_cleaned;

-- 步骤4: 删除不可能再入院的患者
-- 根据业务需求，删除去世或临终关怀的患者
-- 定义代表去世或临终关怀的discharge_disposition_id: [11, 13, 14, 19, 20, 21]
-- 11: Expired, 13: Hospice / home, 14: Hospice / medical facility, 19: Expired at home, 20: Expired in medical facility, 21: Expired place unknown

-- 创建最终业务清洗表
CREATE TABLE IF NOT EXISTS patients_business_cleaned AS
SELECT *
FROM patients_final_cleaned
WHERE discharge_disposition_desc NOT IN (
    'Expired',
    'Hospice / home', 
    'Hospice / medical facility',
    'Expired at home. Medicaid only, hospice.',
    'Expired in a medical facility. Medicaid only, hospice.',
    'Expired, place unknown. Medicaid only, hospice.'
);

-- 创建索引以提高查询性能
CREATE INDEX idx_patient_id_cleaned ON patients_business_cleaned(patient_id);
CREATE INDEX idx_readmitted_cleaned ON patients_business_cleaned(readmitted);
CREATE INDEX idx_age_cleaned ON patients_business_cleaned(age);
CREATE INDEX idx_time_in_hospital_cleaned ON patients_business_cleaned(time_in_hospital);

-- 显示清洗结果统计
SELECT 
    'Original' as stage,
    COUNT(*) as record_count
FROM patients_mapped
UNION ALL
SELECT 
    'After ID Removal' as stage,
    COUNT(*) as record_count
FROM patients_cleaned
UNION ALL
SELECT 
    'After Invalid Value Analysis' as stage,
    COUNT(*) as record_count
FROM patients_final_cleaned
UNION ALL
SELECT 
    'After Business Rules' as stage,
    COUNT(*) as record_count
FROM patients_business_cleaned;

-- 显示列删除统计
SELECT 
    'Column Removal Summary' as summary,
    (SELECT COUNT(*) FROM invalid_value_analysis WHERE invalid_percentage > 50) as columns_removed,
    (SELECT COUNT(*) FROM invalid_value_analysis WHERE invalid_percentage <= 50) as columns_kept,
    (SELECT COUNT(*) FROM invalid_value_analysis) as total_columns_analyzed;

-- 显示被删除的患者统计
SELECT 
    'Removed Records' as category,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM patients_mapped), 2) as percentage
FROM patients_final_cleaned
WHERE discharge_disposition_desc IN (
    'Expired',
    'Hospice / home', 
    'Hospice / medical facility',
    'Expired at home. Medicaid only, hospice.',
    'Expired in a medical facility. Medicaid only, hospice.',
    'Expired, place unknown. Medicaid only, hospice.'
); 