"""
Configuration Management Utility
Handles loading and managing configuration files
"""

import os
import yaml
from typing import Dict, Any
import logging

class ConfigManager:
    """Configuration manager for the hospital readmission project"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = config_dir
        self.configs = {}
        self._load_configs()
    
    def _load_configs(self):
        """Load all configuration files"""
        config_files = [
            "database_config.yaml",
            "azure_config.yaml"
        ]
        
        for config_file in config_files:
            config_path = os.path.join(self.config_dir, config_file)
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.configs[config_file.replace('.yaml', '')] = yaml.safe_load(f)
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return self.configs.get('database_config', {}).get('database', {})
    
    def get_azure_config(self) -> Dict[str, Any]:
        """Get Azure configuration"""
        return self.configs.get('azure_config', {}).get('azure', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.configs.get('database_config', {}).get('logging', {})
    
    def get_data_processing_config(self) -> Dict[str, Any]:
        """Get data processing configuration"""
        return self.configs.get('database_config', {}).get('data_processing', {})
    
    def get_mysql_config(self) -> Dict[str, Any]:
        """Get MySQL specific configuration"""
        return self.get_database_config().get('mysql', {})
    
    def get_environment_config(self, env: str = 'development') -> Dict[str, Any]:
        """Get environment specific configuration"""
        return self.configs.get('azure_config', {}).get('environment', {}).get(env, {}) 