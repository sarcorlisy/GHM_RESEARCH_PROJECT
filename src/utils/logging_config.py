"""
Logging Configuration Utility
Sets up structured logging for the entire pipeline
"""

import logging
import os
from datetime import datetime
from typing import Optional

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """
    Set up structured logging for the pipeline
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        log_format: Log message format
        
    Returns:
        Configured logger instance
    """
    
    # Create logs directory if it doesn't exist
    if log_file and not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),  # Console handler
        ]
    )
    
    # Add file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
    
    # Create and return logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized with level: {log_level}")
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

class PipelineLogger:
    """Context manager for pipeline logging"""
    
    def __init__(self, pipeline_name: str, log_file: Optional[str] = None):
        self.pipeline_name = pipeline_name
        self.log_file = log_file or f"logs/{pipeline_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.logger = None
    
    def __enter__(self):
        self.logger = setup_logging(log_file=self.log_file)
        self.logger.info(f"🚀 Starting pipeline: {self.pipeline_name}")
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.logger.error(f"❌ Pipeline failed: {exc_val}")
        else:
            self.logger.info(f"✅ Pipeline completed successfully: {self.pipeline_name}")
        return False  # Don't suppress exceptions 