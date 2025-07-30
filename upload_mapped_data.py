#!/usr/bin/env python3
"""
上传mapped数据到Azure Data Lake
将经过mapping处理的完整数据上传到processed-data容器
"""

import pandas as pd
import mysql.connector
from mysql.connector import Error
import logging
from datetime import datetime
import os

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MappedDataUploader:
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
                password=self.password
            )
            if self.connection.is_connected():
                logger.info(f"✅ 成功连接到数据库: {self.database}")
                return True
        except Error as e:
            logger.error(f"❌ 连接数据库失败: {e}")
            return False
    
    def disconnect(self):
        """断开数据库连接"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("数据库连接已关闭")
    
    def get_mapped_data(self):
        """获取mapped数据"""
        try:
            query = """
            SELECT 
                patient_id,
                encounter_id,
                race,
                gender,
                age,
                weight,
                admission_type_id,
                admission_type_desc,
                discharge_disposition_id,
                discharge_disposition_desc,
                admission_source_id,
                admission_source_desc,
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
                readmitted
            FROM patients_mapped
            WHERE patient_id IS NOT NULL
            """
            
            df = pd.read_sql(query, self.connection)
            logger.info(f"📊 获取mapped数据: {len(df)} 行, {len(df.columns)} 列")
            return df
            
        except Error as e:
            logger.error(f"❌ 获取mapped数据失败: {e}")
            return None
    
    def save_to_csv(self, df, filename):
        """保存数据到CSV文件"""
        try:
            df.to_csv(filename, index=False)
            logger.info(f"💾 数据已保存到: {filename}")
            return True
        except Exception as e:
            logger.error(f"❌ 保存CSV失败: {e}")
            return False
    
    def upload_to_azure(self, local_file, container_name="processed-data", blob_name=None):
        """上传文件到Azure"""
        try:
            # 导入Azure相关模块
            from azure.storage.blob import BlobServiceClient
            import os
            
            # 获取连接字符串
            connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
            
            if not connection_string:
                logger.error("❌ 未找到AZURE_STORAGE_CONNECTION_STRING环境变量")
                return False
            
            # 创建blob客户端
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            container_client = blob_service_client.get_container_client(container_name)
            
            # 生成blob名称
            if not blob_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                blob_name = f"mapped_data_{timestamp}.csv"
            
            # 上传文件
            with open(local_file, "rb") as data:
                blob_client = container_client.get_blob_client(blob_name)
                blob_client.upload_blob(data, overwrite=True)
            
            logger.info(f"✅ 成功上传到Azure: {container_name}/{blob_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 上传Azure失败: {e}")
            return False

def main():
    """主函数"""
    print("🚀 Mapped数据上传工具")
    print("=" * 60)
    
    # 创建上传器
    uploader = MappedDataUploader()
    
    # 连接数据库
    if not uploader.connect():
        return
    
    try:
        # 获取mapped数据
        df = uploader.get_mapped_data()
        if df is None:
            return
        
        # 保存到本地CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        local_file = f"mapped_data_{timestamp}.csv"
        
        if uploader.save_to_csv(df, local_file):
            # 上传到Azure
            if uploader.upload_to_azure(local_file, blob_name="mapped_data.csv"):
                print("🎉 Mapped数据上传成功！")
                print(f"📊 数据统计: {len(df)} 行, {len(df.columns)} 列")
                print(f"📁 本地文件: {local_file}")
                print(f"☁️ Azure位置: processed-data/mapped_data.csv")
            else:
                print("❌ Azure上传失败")
        
        # 清理本地文件
        try:
            os.remove(local_file)
            logger.info(f"🗑️ 清理本地文件: {local_file}")
        except:
            pass
            
    finally:
        uploader.disconnect()

if __name__ == "__main__":
    main() 