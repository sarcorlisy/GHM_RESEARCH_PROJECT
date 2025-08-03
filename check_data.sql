-- 检查当前数据库
SELECT DATABASE() as current_database;

-- 检查patients表是否存在
SHOW TABLES LIKE 'patients';

-- 检查patients表结构
DESCRIBE patients;

-- 检查patients表记录数
SELECT COUNT(*) as total_records FROM patients;

-- 查看前5条记录
SELECT * FROM patients LIMIT 5;

-- 检查所有表
SHOW TABLES; 