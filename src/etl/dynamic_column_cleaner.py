"""
Dynamic Column Cleaner
Based on invalid value analysis results, actually delete columns with invalid value percentage over 50%
"""

import mysql.connector
import yaml
from typing import List, Dict, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamicColumnCleaner:
    def __init__(self, config_path: str = "config/database_config.yaml"):
        """Initialize dynamic column cleaner"""
        self.config = self._load_config(config_path)
        self.connection = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load database configuration"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Failed to load configuration file: {e}")
            raise
    
    def connect_database(self):
        """Connect to database"""
        try:
            # Get MySQL configuration from config
            mysql_config = self.config['database']['mysql']
            self.connection = mysql.connector.connect(
                host=mysql_config['host'],
                user=mysql_config['user'],
                password=mysql_config['password'],
                database=mysql_config['database'],
                port=mysql_config.get('port', 3306)
            )
            logger.info("Database connection successful")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def get_invalid_value_analysis(self) -> List[Dict]:
        """Get invalid value analysis results"""
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            # First check if invalid_value_analysis table exists
            cursor.execute("""
            SELECT COUNT(*) as table_exists 
            FROM information_schema.tables 
            WHERE table_schema = DATABASE() 
            AND table_name = 'invalid_value_analysis'
            """)
            result = cursor.fetchone()
            
            if result['table_exists'] == 0:
                # Table doesn't exist, automatically create and populate data
                logger.info("invalid_value_analysis table doesn't exist, creating and populating data...")
                
                # Check if source table exists
                cursor.execute("""
                SELECT COUNT(*) as table_exists 
                FROM information_schema.tables 
                WHERE table_schema = DATABASE() 
                AND table_name = 'patients_mapped'
                """)
                source_exists = cursor.fetchone()
                
                if source_exists['table_exists'] == 0:
                    logger.warning("Source table patients_mapped doesn't exist, cannot create invalid value analysis table")
                    cursor.close()
                    return []
                
                # Get all columns in the table
                cursor.execute("""
                SELECT COLUMN_NAME 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_NAME = 'patients_mapped' 
                AND TABLE_SCHEMA = DATABASE()
                ORDER BY ORDINAL_POSITION
                """)
                all_columns = [row['COLUMN_NAME'] for row in cursor.fetchall()]
                
                logger.info(f"📊 Analyzing {len(all_columns)} columns: {', '.join(all_columns)}")
                
                # Generate analysis SQL for each column
                analysis_queries = []
                for column in all_columns:
                    analysis_query = f"""
                    SELECT 
                        '{column}' as column_name,
                        COUNT(CASE WHEN `{column}` IN ('Unknown', 'Not Available', 'NULL', '') OR `{column}` IS NULL THEN 1 END) as invalid_count,
                        COUNT(*) as total_count,
                        ROUND(COUNT(CASE WHEN `{column}` IN ('Unknown', 'Not Available', 'NULL', '') OR `{column}` IS NULL THEN 1 END) * 100.0 / COUNT(*), 2) as invalid_percentage
                    FROM patients_mapped
                    """
                    analysis_queries.append(analysis_query)
                
                # Combine all queries
                combined_query = " UNION ALL ".join(analysis_queries)
                create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS invalid_value_analysis AS
                {combined_query}
                """
                
                # Execute table creation
                cursor.execute(create_table_sql)
                self.connection.commit()
                logger.info(" invalid_value_analysis table created and populated successfully")
            
            # Query invalid value analysis results
            query = """
            SELECT 
                column_name,
                invalid_count,
                total_count,
                invalid_percentage
            FROM invalid_value_analysis 
            ORDER BY invalid_percentage DESC
            """
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            
            # Print analysis results
            logger.info("📋 Invalid value analysis results:")
            for result in results:
                logger.info(f"  - {result['column_name']}: {result['invalid_percentage']}% invalid values ({result['invalid_count']}/{result['total_count']})")
            
            return results
                
        except Exception as e:
            logger.error(f"Failed to get invalid value analysis: {e}")
            # Return default empty results to avoid program crash
            return []
    
    def get_table_columns(self, table_name: str) -> List[str]:
        """Get table column names"""
        try:
            cursor = self.connection.cursor()
            query = f"""
            SELECT COLUMN_NAME 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = '{table_name}' 
            AND TABLE_SCHEMA = DATABASE()
            ORDER BY ORDINAL_POSITION
            """
            cursor.execute(query)
            columns = [row[0] for row in cursor.fetchall()]
            cursor.close()
            return columns
        except Exception as e:
            logger.error(f"Failed to get table column names: {e}")
            raise
    
    def build_dynamic_sql(self, source_table: str, target_table: str, 
                         columns_to_keep: List[str]) -> str:
        """Build dynamic SQL statement - fix data type conversion issues"""
        try:
            # Get all columns from source table
            all_columns = self.get_table_columns(source_table)
            
            # Filter columns to keep
            columns_to_select = []
            for col in all_columns:
                if col in columns_to_keep:
                    columns_to_select.append(col)
            
            # Build SQL statement, handle data type conversion
            column_definitions = []
            for col in columns_to_select:
                # For numeric columns that may contain non-numeric values, use CASE statement
                if col in ['age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                          'num_medications', 'number_outpatient', 'number_emergency', 
                          'number_inpatient', 'number_diagnoses', 'admission_type_id',
                          'discharge_disposition_id', 'admission_source_id']:
                    column_definitions.append(f"""
                        CASE 
                            WHEN `{col}` IN ('Unknown', 'Not Available', 'NULL', '') OR `{col}` IS NULL 
                            THEN NULL 
                            ELSE CAST(`{col}` AS SIGNED) 
                        END as `{col}`
                    """)
                else:
                    # For string columns, select directly
                    column_definitions.append(f"`{col}`")
            
            columns_str = ', '.join(column_definitions)
            sql = f"CREATE TABLE IF NOT EXISTS {target_table} AS SELECT {columns_str} FROM {source_table}"
            return sql
        except Exception as e:
            logger.error(f"Failed to build dynamic SQL: {e}")
            raise
    
    def execute_dynamic_cleaning(self, source_table: str = "patients_mapped", 
                               target_table: str = "patients_cleaned",
                               invalid_threshold: float = 50.0) -> Dict:
        """Execute dynamic column cleaning - fixed version"""
        try:
            logger.info("Starting dynamic column cleaning...")
            
            # Set auto-commit to avoid transaction issues
            self.connection.autocommit = True
            
            # Get invalid value analysis results
            analysis_results = self.get_invalid_value_analysis()
            
            # Determine columns to delete
            columns_to_remove = []
            for result in analysis_results:
                column_name = result['column_name']
                invalid_percentage = result['invalid_percentage']
                
                if invalid_percentage > invalid_threshold:
                    columns_to_remove.append(column_name)
            
            # Get all columns from source table
            all_columns = self.get_table_columns(source_table)
            
            # Keep all columns not in deletion list
            columns_to_keep = [col for col in all_columns if col not in columns_to_remove]
            
            logger.info(f"Columns to keep: {columns_to_keep}")
            logger.info(f"Columns to delete: {columns_to_remove}")
            
            # Build and execute dynamic SQL
            dynamic_sql = self.build_dynamic_sql(source_table, target_table, columns_to_keep)
            
            # Print generated SQL statement for debugging
            logger.info(f"Generated SQL statement: {dynamic_sql}")
            
            # Use retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    cursor = self.connection.cursor()
                    
                    # Set session timeout
                    cursor.execute("SET SESSION wait_timeout = 300")
                    cursor.execute("SET SESSION interactive_timeout = 300")
                    
                    logger.info(f"Starting SQL execution (attempt {attempt + 1}/{max_retries}), this may take several minutes...")
                    cursor.execute(dynamic_sql)
                    cursor.close()
                    logger.info("SQL execution completed")
                    break
                    
                except mysql.connector.Error as e:
                    if 'Table definition has changed' in str(e) and attempt < max_retries - 1:
                        logger.warning(f"Table definition change error, waiting before retry: {e}")
                        import time
                        time.sleep(2)  # Wait 2 seconds before retry
                        continue
                    else:
                        raise
            
            # Verify results
            final_columns = self.get_table_columns(target_table)
            
            result = {
                'source_table': source_table,
                'target_table': target_table,
                'columns_kept': columns_to_keep,
                'columns_removed': columns_to_remove,
                'final_columns': final_columns,
                'total_columns_kept': len(columns_to_keep),
                'total_columns_removed': len(columns_to_remove)
            }
            
            logger.info(f"Dynamic column cleaning completed! Kept {len(columns_to_keep)} columns, deleted {len(columns_to_remove)} columns")
            return result
            
        except Exception as e:
            logger.error(f"Dynamic column cleaning failed: {e}")
            raise
    
    def close_connection(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

def main():
    """Main function"""
    try:
        # Create dynamic column cleaner
        cleaner = DynamicColumnCleaner()
        
        # Connect to database
        cleaner.connect_database()
        
        # Execute dynamic column cleaning
        result = cleaner.execute_dynamic_cleaning()
        
        # Print results
        print("\n=== Dynamic Column Cleaning Results ===")
        print(f"Source table: {result['source_table']}")
        print(f"Target table: {result['target_table']}")
        print(f"Columns kept: {result['total_columns_kept']}")
        print(f"Columns deleted: {result['total_columns_removed']}")
        print(f"\nDeleted columns: {result['columns_removed']}")
        print(f"\nKept columns: {result['columns_kept']}")
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")
    finally:
        cleaner.close_connection()

if __name__ == "__main__":
    main() 