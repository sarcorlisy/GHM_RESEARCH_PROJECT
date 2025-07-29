"""
Database Extension Module for Hospital Readmission Prediction Pipeline

This module provides database connectivity and operations for the analytics pipeline.
It extends the existing pipeline with PostgreSQL database capabilities.

Modules:
- db_connector: Database connection and management
- sql_queries: Predefined SQL queries for analytics
- data_migration: CSV to database migration utilities
"""

from .db_connector import DatabaseConnector, DatabaseManager
from .sql_queries import HospitalReadmissionQueries, QueryExecutor
from .data_migration import DataValidator, DataTransformer, DataMigrator

__all__ = [
    'DatabaseConnector',
    'DatabaseManager', 
    'HospitalReadmissionQueries',
    'QueryExecutor',
    'DataValidator',
    'DataTransformer',
    'DataMigrator'
]

__version__ = '1.0.0'
__author__ = 'Analytics Engineer'
__description__ = 'Database extension for hospital readmission prediction pipeline' 