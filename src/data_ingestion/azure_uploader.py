"""
Azure Data Upload Module
Handles uploading raw data to Azure Data Lake Storage
"""

import os
import pandas as pd
from typing import Optional, Dict, Any
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
import logging

from ..utils.config import ConfigManager
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

class AzureDataUploader:
    """Handles data upload to Azure Data Lake Storage"""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        self.config = config or ConfigManager()
        self.azure_config = self.config.get_azure_config()
        self.blob_service_client = None
        self._initialize_azure_client()
    
    def _initialize_azure_client(self):
        """Initialize Azure Blob Service Client"""
        try:
            # Try to use connection string first
            connection_string = self.azure_config.get('storage_account', {}).get('connection_string')
            if connection_string and connection_string != 'your_connection_string':
                self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                logger.info("✅ Connected to Azure Storage using connection string")
            else:
                # Fall back to default credential (for development)
                logger.warning("⚠️ Using default Azure credential (for development)")
                self.blob_service_client = BlobServiceClient(
                    account_url=f"https://{self.azure_config.get('storage_account', {}).get('name', 'hospitalreadmissionstorage')}.blob.core.windows.net/",
                    credential=DefaultAzureCredential()
                )
        except Exception as e:
            logger.error(f"❌ Failed to initialize Azure client: {e}")
            logger.info("💡 For development, you can continue without Azure upload")
    
    def upload_csv_to_azure(self, 
                           local_file_path: str, 
                           container_name: str = "raw-data",
                           blob_name: Optional[str] = None) -> bool:
        """
        Upload CSV file to Azure Data Lake Storage
        
        Args:
            local_file_path: Path to local CSV file
            container_name: Azure container name
            blob_name: Name for the blob in Azure (optional)
            
        Returns:
            bool: True if upload successful, False otherwise
        """
        if not self.blob_service_client:
            logger.warning("⚠️ Azure client not initialized, skipping upload")
            return False
        
        try:
            # Generate blob name if not provided
            if not blob_name:
                blob_name = os.path.basename(local_file_path)
            
            # Get container client
            container_client = self.blob_service_client.get_container_client(container_name)
            
            # Upload file
            with open(local_file_path, "rb") as data:
                blob_client = container_client.get_blob_client(blob_name)
                blob_client.upload_blob(data, overwrite=True)
            
            logger.info(f"✅ Successfully uploaded {local_file_path} to Azure as {blob_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to upload {local_file_path}: {e}")
            return False
    
    def upload_dataframe_to_azure(self, 
                                 df: pd.DataFrame,
                                 container_name: str,
                                 blob_name: str,
                                 file_format: str = "csv") -> bool:
        """
        Upload pandas DataFrame to Azure Data Lake Storage
        
        Args:
            df: Pandas DataFrame to upload
            container_name: Azure container name
            blob_name: Name for the blob in Azure
            file_format: File format (csv, parquet, json)
            
        Returns:
            bool: True if upload successful, False otherwise
        """
        if not self.blob_service_client:
            logger.warning("⚠️ Azure client not initialized, skipping upload")
            return False
        
        try:
            # Get container client
            container_client = self.blob_service_client.get_container_client(container_name)
            blob_client = container_client.get_blob_client(blob_name)
            
            # Convert DataFrame to bytes based on format
            if file_format.lower() == "csv":
                data = df.to_csv(index=False).encode('utf-8')
            elif file_format.lower() == "parquet":
                data = df.to_parquet(index=False)
            elif file_format.lower() == "json":
                data = df.to_json(orient='records').encode('utf-8')
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            # Upload data
            blob_client.upload_blob(data, overwrite=True)
            
            logger.info(f"✅ Successfully uploaded DataFrame to Azure as {blob_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to upload DataFrame: {e}")
            return False
    
    def list_blobs_in_container(self, container_name: str) -> list:
        """
        List all blobs in a container
        
        Args:
            container_name: Azure container name
            
        Returns:
            list: List of blob names
        """
        if not self.blob_service_client:
            logger.warning("⚠️ Azure client not initialized")
            return []
        
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            blobs = [blob.name for blob in container_client.list_blobs()]
            logger.info(f"📋 Found {len(blobs)} blobs in container {container_name}")
            return blobs
        except Exception as e:
            logger.error(f"❌ Failed to list blobs: {e}")
            return []
    
    def download_blob_from_azure(self, 
                                container_name: str,
                                blob_name: str,
                                local_file_path: str) -> bool:
        """
        Download blob from Azure Data Lake Storage
        
        Args:
            container_name: Azure container name
            blob_name: Name of the blob in Azure
            local_file_path: Local path to save the file
            
        Returns:
            bool: True if download successful, False otherwise
        """
        if not self.blob_service_client:
            logger.warning("⚠️ Azure client not initialized")
            return False
        
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            blob_client = container_client.get_blob_client(blob_name)
            
            with open(local_file_path, "wb") as download_file:
                download_stream = blob_client.download_blob()
                download_file.write(download_stream.readall())
            
            logger.info(f"✅ Successfully downloaded {blob_name} to {local_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download {blob_name}: {e}")
            return False 