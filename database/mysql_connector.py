"""
MySQL数据库连接器

专门为MySQL数据库设计的连接器和查询执行器
"""

import pandas as pd
import mysql.connector
from mysql.connector import Error
from sqlalchemy import create_engine, text
import os
from datetime import datetime
import logging

class MySQLConnector:
    """MySQL数据库连接器"""
    
    def __init__(self, config=None):
        """
        初始化MySQL连接器
        
        Args:
            config (dict): 数据库配置
        """
        self.config = config or self._get_default_config()
        self.connection = None
        self.engine = None
        self._setup_logging()
    
    def _get_default_config(self):
        """获取默认配置"""
        return {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '3306')),
            'database': os.getenv('DB_NAME', 'hospital_readmission'),
            'user': os.getenv('DB_USER', 'root'),
            'password': os.getenv('DB_PASSWORD', ''),
            'charset': 'utf8mb4',
            'autocommit': True
        }
    
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def connect(self):
        """连接到MySQL数据库"""
        try:
            # 创建数据库连接
            self.connection = mysql.connector.connect(**self.config)
            
            # 创建SQLAlchemy引擎
            connection_string = (
                f"mysql+mysqlconnector://{self.config['user']}:{self.config['password']}"
                f"@{self.config['host']}:{self.config['port']}/{self.config['database']}"
            )
            self.engine = create_engine(connection_string)
            
            self.logger.info(f"✅ 成功连接到MySQL数据库: {self.config['host']}:{self.config['port']}")
            return True
            
        except Error as e:
            self.logger.error(f"❌ MySQL连接失败: {e}")
            return False
    
    def disconnect(self):
        """断开数据库连接"""
        if self.connection:
            self.connection.close()
            self.logger.info("数据库连接已关闭")
    
    def create_database_if_not_exists(self):
        """创建数据库（如果不存在）"""
        try:
            # 临时连接（不指定数据库）
            temp_config = self.config.copy()
            temp_config.pop('database', None)
            
            temp_connection = mysql.connector.connect(**temp_config)
            cursor = temp_connection.cursor()
            
            # 创建数据库
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.config['database']} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            self.logger.info(f"✅ 数据库 {self.config['database']} 已创建或已存在")
            
            cursor.close()
            temp_connection.close()
            
            return True
            
        except Error as e:
            self.logger.error(f"❌ 创建数据库失败: {e}")
            return False
    
    def create_tables(self):
        """创建必要的表"""
        try:
            if not self.connection:
                self.connect()
            
            cursor = self.connection.cursor()
            
            # 创建患者表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patients (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    patient_id VARCHAR(50) UNIQUE NOT NULL,
                    age INT,
                    gender VARCHAR(10),
                    race VARCHAR(50),
                    admission_type_id INT,
                    discharge_disposition_id INT,
                    admission_source_id INT,
                    time_in_hospital INT,
                    num_lab_procedures INT,
                    num_procedures INT,
                    num_medications INT,
                    number_outpatient INT,
                    number_emergency INT,
                    number_inpatient INT,
                    diag_1 VARCHAR(10),
                    diag_2 VARCHAR(10),
                    diag_3 VARCHAR(10),
                    readmitted VARCHAR(10),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_patient_id (patient_id),
                    INDEX idx_readmitted (readmitted),
                    INDEX idx_age (age),
                    INDEX idx_time_in_hospital (time_in_hospital)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            
            # 创建就诊记录表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS encounters (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    encounter_id VARCHAR(50) UNIQUE NOT NULL,
                    patient_id VARCHAR(50),
                    admission_date DATE,
                    discharge_date DATE,
                    length_of_stay INT,
                    readmission_within_30_days BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
                    INDEX idx_encounter_id (encounter_id),
                    INDEX idx_patient_id (patient_id),
                    INDEX idx_admission_date (admission_date)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            
            # 创建药物信息表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS medications (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    patient_id VARCHAR(50),
                    medication_name VARCHAR(100),
                    dosage VARCHAR(50),
                    frequency VARCHAR(50),
                    start_date DATE,
                    end_date DATE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
                    INDEX idx_patient_id (patient_id),
                    INDEX idx_medication_name (medication_name)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            
            # 创建模型结果表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_results (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    model_name VARCHAR(100),
                    feature_selection_method VARCHAR(50),
                    test_accuracy DECIMAL(5,4),
                    test_precision DECIMAL(5,4),
                    test_recall DECIMAL(5,4),
                    test_f1_score DECIMAL(5,4),
                    test_auc DECIMAL(5,4),
                    cv_accuracy DECIMAL(5,4),
                    cv_std DECIMAL(5,4),
                    training_time_seconds DECIMAL(10,2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_model_name (model_name),
                    INDEX idx_created_at (created_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            
            cursor.close()
            self.logger.info("✅ 所有表创建成功")
            return True
            
        except Error as e:
            self.logger.error(f"❌ 创建表失败: {e}")
            return False
    
    def insert_dataframe(self, df, table_name, if_exists='append'):
        """将DataFrame插入到MySQL表"""
        try:
            if not self.engine:
                self.connect()
            
            # 使用SQLAlchemy插入数据
            df.to_sql(
                name=table_name,
                con=self.engine,
                if_exists=if_exists,
                index=False,
                method='multi',
                chunksize=1000
            )
            
            self.logger.info(f"✅ 成功插入 {len(df)} 行数据到表 {table_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 插入数据失败: {e}")
            return False
    
    def execute_query(self, query, params=None):
        """执行SQL查询并返回DataFrame"""
        try:
            if not self.connection:
                self.connect()
            
            # 使用pandas读取SQL查询结果
            df = pd.read_sql(query, self.engine, params=params)
            return df
            
        except Exception as e:
            self.logger.error(f"❌ 执行查询失败: {e}")
            return pd.DataFrame()
    
    def execute_update(self, query, params=None):
        """执行更新操作"""
        try:
            if not self.connection:
                self.connect()
            
            cursor = self.connection.cursor()
            cursor.execute(query, params or ())
            self.connection.commit()
            
            affected_rows = cursor.rowcount
            cursor.close()
            
            self.logger.info(f"✅ 更新操作成功，影响 {affected_rows} 行")
            return affected_rows
            
        except Error as e:
            self.logger.error(f"❌ 更新操作失败: {e}")
            return 0
    
    def get_table_info(self, table_name):
        """获取表信息"""
        try:
            query = f"""
                SELECT 
                    COLUMN_NAME,
                    DATA_TYPE,
                    IS_NULLABLE,
                    COLUMN_DEFAULT,
                    COLUMN_COMMENT
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = '{self.config['database']}' 
                AND TABLE_NAME = '{table_name}'
                ORDER BY ORDINAL_POSITION
            """
            return self.execute_query(query)
            
        except Exception as e:
            self.logger.error(f"❌ 获取表信息失败: {e}")
            return pd.DataFrame()
    
    def get_table_count(self, table_name):
        """获取表的行数"""
        try:
            query = f"SELECT COUNT(*) as count FROM {table_name}"
            result = self.execute_query(query)
            return result['count'].iloc[0] if not result.empty else 0
            
        except Exception as e:
            self.logger.error(f"❌ 获取表行数失败: {e}")
            return 0

class MySQLManager:
    """MySQL数据库管理器"""
    
    def __init__(self, config=None):
        self.connector = MySQLConnector(config)
    
    def initialize_database(self):
        """初始化数据库"""
        try:
            # 创建数据库
            if not self.connector.create_database_if_not_exists():
                return False
            
            # 连接数据库
            if not self.connector.connect():
                return False
            
            # 创建表
            if not self.connector.create_tables():
                return False
            
            self.connector.logger.info("✅ MySQL数据库初始化完成")
            return True
            
        except Exception as e:
            self.connector.logger.error(f"❌ 数据库初始化失败: {e}")
            return False
    
    def migrate_csv_to_database(self, csv_file, table_name, chunk_size=1000):
        """将CSV文件迁移到数据库"""
        try:
            if not os.path.exists(csv_file):
                self.connector.logger.error(f"❌ CSV文件不存在: {csv_file}")
                return False
            
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            self.connector.logger.info(f"📁 读取CSV文件: {csv_file} ({len(df)} 行)")
            
            # 插入数据
            success = self.connector.insert_dataframe(df, table_name)
            
            if success:
                count = self.connector.get_table_count(table_name)
                self.connector.logger.info(f"✅ 数据迁移完成，表 {table_name} 现在有 {count} 行数据")
            
            return success
            
        except Exception as e:
            self.connector.logger.error(f"❌ CSV迁移失败: {e}")
            return False
    
    def get_database_summary(self):
        """获取数据库摘要信息"""
        try:
            tables = ['patients', 'encounters', 'medications', 'model_results']
            summary = {}
            
            for table in tables:
                count = self.connector.get_table_count(table)
                summary[table] = count
            
            return summary
            
        except Exception as e:
            self.connector.logger.error(f"❌ 获取数据库摘要失败: {e}")
            return {}
    
    def close(self):
        """关闭数据库连接"""
        self.connector.disconnect() 