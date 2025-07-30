"""
数据导入脚本 - 将Azure数据导入MySQL数据库

从Azure下载原始数据，应用首次入院逻辑，然后导入MySQL
"""

import pandas as pd
import mysql.connector
from mysql.connector import Error
import logging
import os
import tempfile

# 添加Azure相关导入
try:
    from azure.storage.blob import BlobServiceClient
    from azure.identity import DefaultAzureCredential
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    print("⚠️ Azure SDK not available, will use local files")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataImporter:
    def __init__(self, host='localhost', database='hospital_readmission', 
                 user='root', password='hospital123'):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.connection = None
        
    def connect(self):
        """连接到MySQL数据库"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                charset='utf8mb4'
            )
            logger.info(f"✅ 成功连接到数据库: {self.database}")
            return True
        except Error as e:
            logger.error(f"❌ 连接失败: {e}")
            return False
    
    def disconnect(self):
        """断开数据库连接"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("数据库连接已关闭")
    
    def download_from_azure(self, container_name="raw-data", blob_name="diabetic_data.csv"):
        """
        从Azure下载原始数据文件
        
        Args:
            container_name: Azure容器名称
            blob_name: Azure blob名称
            
        Returns:
            str: 下载文件的本地路径，如果失败返回None
        """
        if not AZURE_AVAILABLE:
            logger.warning("⚠️ Azure SDK不可用，将使用本地文件")
            return None
            
        try:
            # 尝试使用连接字符串（从环境变量获取）
            connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
            
            if connection_string:
                blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                logger.info("✅ 使用连接字符串连接到Azure Storage")
            else:
                # 使用默认凭据（开发环境）
                blob_service_client = BlobServiceClient(
                    account_url="https://hospitalreadmissionstorage.blob.core.windows.net/",
                    credential=DefaultAzureCredential()
                )
                logger.info("✅ 使用默认凭据连接到Azure Storage")
            
            # 获取容器和blob客户端
            container_client = blob_service_client.get_container_client(container_name)
            blob_client = container_client.get_blob_client(blob_name)
            
            # 创建临时文件
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            temp_file_path = temp_file.name
            temp_file.close()
            
            # 下载文件
            logger.info(f"📥 从Azure下载文件: {container_name}/{blob_name}")
            with open(temp_file_path, "wb") as download_file:
                download_stream = blob_client.download_blob()
                download_file.write(download_stream.readall())
            
            logger.info(f"✅ 文件已下载到: {temp_file_path}")
            return temp_file_path
            
        except Exception as e:
            logger.error(f"❌ 从Azure下载文件失败: {e}")
            logger.info("💡 将尝试使用本地文件")
            return None
    
    def clear_tables(self):
        """清空所有表（按正确顺序处理外键约束）"""
        try:
            cursor = self.connection.cursor()
            
            # 按顺序删除表（先删除有外键引用的表）
            tables_to_drop = [
                'model_results',  # 没有外键引用
                'medications',    # 可能引用encounters
                'encounters',     # 引用patients
                'patients'        # 被其他表引用
            ]
            
            for table in tables_to_drop:
                cursor.execute(f"DROP TABLE IF EXISTS {table}")
                logger.info(f"🗑️ 删除表: {table}")
            
            self.connection.commit()
            cursor.close()
            logger.info("✅ 所有表已清空")
            return True
            
        except Error as e:
            logger.error(f"❌ 清空表失败: {e}")
            return False
    
    def recreate_tables(self):
        """重新创建表结构"""
        try:
            cursor = self.connection.cursor()
            
            # 创建patients表 - 包含所有50列原始数据
            patients_table = """
            CREATE TABLE patients (
                id INT AUTO_INCREMENT PRIMARY KEY,
                encounter_id VARCHAR(50),
                patient_id VARCHAR(50) UNIQUE NOT NULL,
                race VARCHAR(50),
                gender VARCHAR(20),
                age INT,
                weight VARCHAR(20),
                admission_type_id INT,
                discharge_disposition_id INT,
                admission_source_id INT,
                time_in_hospital INT,
                payer_code VARCHAR(20),
                medical_specialty VARCHAR(100),
                num_lab_procedures INT,
                num_procedures INT,
                num_medications INT,
                number_outpatient INT,
                number_emergency INT,
                number_inpatient INT,
                diag_1 VARCHAR(10),
                diag_2 VARCHAR(10),
                diag_3 VARCHAR(10),
                number_diagnoses INT,
                max_glu_serum VARCHAR(20),
                A1Cresult VARCHAR(20),
                metformin VARCHAR(10),
                repaglinide VARCHAR(10),
                nateglinide VARCHAR(10),
                chlorpropamide VARCHAR(10),
                glimepiride VARCHAR(10),
                acetohexamide VARCHAR(10),
                glipizide VARCHAR(10),
                glyburide VARCHAR(10),
                tolbutamide VARCHAR(10),
                pioglitazone VARCHAR(10),
                rosiglitazone VARCHAR(10),
                acarbose VARCHAR(10),
                miglitol VARCHAR(10),
                troglitazone VARCHAR(10),
                tolazamide VARCHAR(10),
                examide VARCHAR(10),
                citoglipton VARCHAR(10),
                insulin VARCHAR(10),
                `glyburide-metformin` VARCHAR(10),
                `glipizide-metformin` VARCHAR(10),
                `glimepiride-pioglitazone` VARCHAR(10),
                `metformin-rosiglitazone` VARCHAR(10),
                `metformin-pioglitazone` VARCHAR(10),
                medication_change VARCHAR(10),
                diabetesMed VARCHAR(10),
                readmitted VARCHAR(10),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_patient_id (patient_id),
                INDEX idx_encounter_id (encounter_id),
                INDEX idx_readmitted (readmitted),
                INDEX idx_age (age),
                INDEX idx_time_in_hospital (time_in_hospital),
                INDEX idx_number_diagnoses (number_diagnoses),
                INDEX idx_medical_specialty (medical_specialty)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            
            # 创建encounters表
            encounters_table = """
            CREATE TABLE encounters (
                id INT AUTO_INCREMENT PRIMARY KEY,
                encounter_id VARCHAR(50) UNIQUE NOT NULL,
                patient_id VARCHAR(50),
                encounter_date DATE,
                discharge_date DATE,
                length_of_stay INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE,
                INDEX idx_encounter_id (encounter_id),
                INDEX idx_patient_id (patient_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            
            # 创建medications表
            medications_table = """
            CREATE TABLE medications (
                id INT AUTO_INCREMENT PRIMARY KEY,
                encounter_id VARCHAR(50),
                medication_name VARCHAR(100),
                dosage VARCHAR(50),
                frequency VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (encounter_id) REFERENCES encounters(encounter_id) ON DELETE CASCADE,
                INDEX idx_encounter_id (encounter_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            
            # 创建model_results表
            model_results_table = """
            CREATE TABLE model_results (
                id INT AUTO_INCREMENT PRIMARY KEY,
                model_name VARCHAR(50),
                feature_selection_method VARCHAR(50),
                accuracy DECIMAL(5,4),
                precision_score DECIMAL(5,4),
                recall DECIMAL(5,4),
                f1_score DECIMAL(5,4),
                auc_score DECIMAL(5,4),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_model_name (model_name),
                INDEX idx_created_at (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            
            # 执行创建表
            tables = [
                ("patients", patients_table),
                ("encounters", encounters_table),
                ("medications", medications_table),
                ("model_results", model_results_table)
            ]
            
            for table_name, table_sql in tables:
                cursor.execute(table_sql)
                logger.info(f"✅ 创建表: {table_name}")
            
            self.connection.commit()
            cursor.close()
            logger.info("✅ 所有表结构重新创建完成")
            return True
            
        except Error as e:
            logger.error(f"❌ 创建表失败: {e}")
            return False
    
    def import_csv_data(self, csv_file='diabetic_data.csv', sample_size=None, use_azure=True):
        """
        导入数据到patients表，严格按照首次入院业务逻辑
        
        Args:
            csv_file: 本地CSV文件路径（备用）
            sample_size: 采样大小（None表示全部数据）
            use_azure: 是否优先从Azure下载数据
        """
        try:
            file_to_use = csv_file
            
            # 优先从Azure下载数据
            if use_azure:
                logger.info("☁️ 尝试从Azure下载原始数据...")
                azure_file = self.download_from_azure()
                if azure_file:
                    file_to_use = azure_file
                    logger.info(f"✅ 使用Azure数据: {azure_file}")
                else:
                    logger.info(f"📁 使用本地文件: {csv_file}")
            
            logger.info(f"📁 开始导入数据文件: {file_to_use}")
            
            # 读取CSV文件
            df = pd.read_csv(file_to_use)
            logger.info(f"📊 原始数据: {len(df)} 行, {len(df.columns)} 列")
            
            # 只保留首次入院记录（notebook逻辑）
            df_sorted = df.sort_values(by='encounter_id')
            df_first = df_sorted.drop_duplicates(subset='patient_nbr', keep='first')
            logger.info(f"📊 首次入院筛选后: {len(df_first)} 行")
            
            # 不再采样，全部导入
            # if sample_size is not None and len(df_first) > sample_size:
            #     df_first = df_first.sample(n=sample_size, random_state=42)
            #     logger.info(f"📊 使用样本数据: {len(df_first)} 行")
            
            # 数据清洗
            df_first = self.clean_data(df_first)
            
            # 准备插入数据
            cursor = self.connection.cursor()
            
            # 构建INSERT语句 - 包含所有50列原始数据
            columns = [
                'encounter_id', 'patient_id', 'race', 'gender', 'age', 'weight',
                'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
                'time_in_hospital', 'payer_code', 'medical_specialty',
                'num_lab_procedures', 'num_procedures', 'num_medications',
                'number_outpatient', 'number_emergency', 'number_inpatient',
                'diag_1', 'diag_2', 'diag_3', 'number_diagnoses',
                'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide',
                'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',
                'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone',
                'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide',
                'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
                'glimepiride-pioglitazone', 'metformin-rosiglitazone',
                'metformin-pioglitazone', 'medication_change', 'diabetesMed', 'readmitted'
            ]
            
            # 重命名列名（如果需要）
            column_mapping = {
                'patient_nbr': 'patient_id',
                'change': 'medication_change'
            }
            df_first = df_first.rename(columns=column_mapping)
            
            # 插入数据 - 处理列名中的连字符
            columns_quoted = [f"`{col}`" for col in columns]
            insert_query = f"""
            INSERT INTO patients ({', '.join(columns_quoted)})
            VALUES ({', '.join(['%s'] * len(columns))})
            """
            
            # 准备数据
            data_to_insert = []
            for _, row in df_first.iterrows():
                row_data = []
                for col in columns:
                    value = row.get(col, None)
                    # 处理特殊值
                    if pd.isna(value) or value == '?':
                        value = None
                    row_data.append(value)
                data_to_insert.append(row_data)
            
            # 批量插入
            cursor.executemany(insert_query, data_to_insert)
            self.connection.commit()
            
            inserted_count = cursor.rowcount
            logger.info(f"✅ 成功插入 {inserted_count} 行数据到patients表")
            
            cursor.close()
            
            # 清理临时文件
            if use_azure and file_to_use != csv_file and os.path.exists(file_to_use):
                try:
                    os.unlink(file_to_use)
                    logger.info(f"🗑️ 清理临时文件: {file_to_use}")
                except Exception as e:
                    logger.warning(f"⚠️ 清理临时文件失败: {e}")
            
            return True
            
        except Error as e:
            logger.error(f"❌ 数据导入失败: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ 处理数据失败: {e}")
            return False
    
    def clean_data(self, df):
        """企业级数据清洗 - 处理所有50列"""
        logger.info("🧹 开始企业级数据清洗...")
        
        # 1. 处理缺失值
        df = df.fillna('Unknown')
        logger.info("✅ 处理缺失值完成")
        
        # 2. 处理特殊字符
        df = df.replace('?', 'Unknown')
        logger.info("✅ 处理特殊字符完成")
        
        # 3. 确保patient_id是字符串
        if 'patient_id' in df.columns:
            df['patient_id'] = df['patient_id'].astype(str)
        
        # 4. 处理年龄字段 - 提取年龄范围的中点
        if 'age' in df.columns:
            def extract_age_midpoint(age_str):
                if pd.isna(age_str) or age_str == 'Unknown':
                    return None
                try:
                    # 处理格式如 '[70-80)' 或 '[0-10)' 等
                    if isinstance(age_str, str) and '[' in age_str and ')' in age_str:
                        # 提取数字范围
                        age_range = age_str.replace('[', '').replace(')', '')
                        if '-' in age_range:
                            start, end = age_range.split('-')
                            return int((int(start) + int(end)) / 2)
                        else:
                            return int(age_range)
                    else:
                        return int(age_str)
                except:
                    return None
            
            df['age'] = df['age'].apply(extract_age_midpoint)
            logger.info("✅ 年龄字段处理完成")
        
        # 5. 处理数值字段 - 确保类型正确
        numeric_columns = [
            'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
            'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
            'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0).astype(int)
        
        logger.info("✅ 数值字段处理完成")
        
        # 6. 处理药物字段 - 标准化
        medication_columns = [
            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
            'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
            'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
            'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
            'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
        ]
        
        for col in medication_columns:
            if col in df.columns:
                df[col] = df[col].str.upper()
                df[col] = df[col].replace(['UNKNOWN', 'NONE'], 'No')
        
        logger.info("✅ 药物字段标准化完成")
        
        # 7. 处理诊断字段 - 清理格式
        diagnosis_columns = ['diag_1', 'diag_2', 'diag_3']
        for col in diagnosis_columns:
            if col in df.columns:
                df[col] = df[col].str.strip()
                df[col] = df[col].replace('', 'Unknown')
        
        logger.info("✅ 诊断字段处理完成")
        
        # 8. 数据质量检查
        total_rows = len(df)
        null_counts = df.isnull().sum()
        logger.info(f"📊 数据清洗完成 - 总行数: {total_rows}")
        logger.info(f"📊 各列空值统计: {null_counts.sum()} 个空值")
        
        return df
    
    def show_table_info(self):
        """显示表信息"""
        try:
            cursor = self.connection.cursor()
            
            tables = ['patients', 'encounters', 'medications', 'model_results']
            
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                logger.info(f"📊 {table}: {count} 行")
            
            cursor.close()
            
        except Error as e:
            logger.error(f"❌ 获取表信息失败: {e}")

def main():
    """主函数"""
    print("🚀 数据导入工具")
    print("=" * 60)
    
    # 创建导入器
    importer = DataImporter()
    
    # 连接数据库
    if not importer.connect():
        return
    
    try:
        # 清空并重新创建表
        print("\n📋 步骤1: 清空现有表")
        if not importer.clear_tables():
            return
        
        print("\n📋 步骤2: 重新创建表结构")
        if not importer.recreate_tables():
            return
        
        # 导入数据
        print("\n📋 步骤3: 导入CSV数据")
        if not importer.import_csv_data():
            return
        
        # 显示结果
        print("\n📋 步骤4: 显示导入结果")
        importer.show_table_info()
        
        print("\n🎉 数据导入完成！")
        print("💡 现在你可以在Navicat中查看数据了")
        
    finally:
        importer.disconnect()

if __name__ == "__main__":
    main() 