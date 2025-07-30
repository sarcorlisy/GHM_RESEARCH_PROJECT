#!/usr/bin/env python3
"""
上传清洗后的数据表到Azure Data Lake
将patients_cleaned和patients_business_cleaned两个表上传到processed-data容器
"""
import pandas as pd
import mysql.connector
from mysql.connector import Error
import logging
from datetime import datetime
import os
from azure.storage.blob import BlobServiceClient
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class CleanedTablesUploader:
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
            logger.info(f"✅ 成功连接到数据库: {self.database}")
            return True
        except Error as e:
            logger.error(f"❌ 数据库连接失败: {e}")
            return False
    
    def disconnect(self):
        """断开数据库连接"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("数据库连接已关闭")
    
    def get_cleaned_data(self, table_name):
        """获取清洗后的数据"""
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, self.connection)
        logger.info(f"📊 获取{table_name}数据: {len(df)} 行, {len(df.columns)} 列")
        return df
    
    def save_to_csv(self, df, filename):
        """保存数据到CSV文件"""
        try:
            df.to_csv(filename, index=False, encoding='utf-8')
            logger.info(f"💾 数据已保存到: {filename}")
            return True
        except Exception as e:
            logger.error(f"❌ 保存CSV失败: {e}")
            return False
    
    def upload_to_azure(self, local_file, container_name="processed-data", blob_name=None):
        """上传文件到Azure"""
        try:
            # 获取Azure连接字符串
            connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
            if not connection_string:
                logger.error("❌ 未找到AZURE_STORAGE_CONNECTION_STRING环境变量")
                return False
            
            # 创建Blob服务客户端
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            
            # 获取容器客户端
            container_client = blob_service_client.get_container_client(container_name)
            
            # 如果没有指定blob名称，使用文件名
            if not blob_name:
                blob_name = os.path.basename(local_file)
            
            # 获取blob客户端
            blob_client = container_client.get_blob_client(blob_name)
            
            # 上传文件
            with open(local_file, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            
            logger.info(f"✅ 成功上传到Azure: {container_name}/{blob_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Azure上传失败: {e}")
            return False

def main():
    """主函数"""
    print("🚀 清洗表上传工具")
    print("=" * 60)
    
    uploader = CleanedTablesUploader()
    if not uploader.connect():
        return
    
    try:
        # 上传第一个表：patients_cleaned
        print("\n📊 上传 patients_cleaned 表...")
        df_cleaned = uploader.get_cleaned_data('patients_cleaned')
        if df_cleaned is None:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        local_file_cleaned = f"patients_cleaned_{timestamp}.csv"
        
        if uploader.save_to_csv(df_cleaned, local_file_cleaned):
            if uploader.upload_to_azure(local_file_cleaned, blob_name="patients_cleaned.csv"):
                print("✅ patients_cleaned 上传成功！")
                print(f"📊 数据统计: {len(df_cleaned)} 行, {len(df_cleaned.columns)} 列")
                print(f"☁️ Azure位置: processed-data/patients_cleaned.csv")
            else:
                print("❌ patients_cleaned Azure上传失败")
        
        # 上传第二个表：patients_business_cleaned
        print("\n📊 上传 patients_business_cleaned 表...")
        df_business = uploader.get_cleaned_data('patients_business_cleaned')
        if df_business is None:
            return
        
        local_file_business = f"patients_business_cleaned_{timestamp}.csv"
        
        if uploader.save_to_csv(df_business, local_file_business):
            if uploader.upload_to_azure(local_file_business, blob_name="patients_business_cleaned.csv"):
                print("✅ patients_business_cleaned 上传成功！")
                print(f"📊 数据统计: {len(df_business)} 行, {len(df_business.columns)} 列")
                print(f"☁️ Azure位置: processed-data/patients_business_cleaned.csv")
            else:
                print("❌ patients_business_cleaned Azure上传失败")
        
        # 清理本地文件
        try:
            os.remove(local_file_cleaned)
            os.remove(local_file_business)
            logger.info(f"🗑️ 清理本地文件: {local_file_cleaned}, {local_file_business}")
        except:
            pass
        
        print("\n🎉 所有清洗表上传完成！")
        print("=" * 60)
        print("📋 上传总结:")
        print(f"   - patients_cleaned: {len(df_cleaned):,} 行, {len(df_cleaned.columns)} 列")
        print(f"   - patients_business_cleaned: {len(df_business):,} 行, {len(df_business.columns)} 列")
        print(f"   - 总记录数: {len(df_cleaned) + len(df_business):,} 行")
            
    finally:
        uploader.disconnect()

if __name__ == "__main__":
    main() 