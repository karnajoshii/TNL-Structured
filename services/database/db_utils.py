import os
import mysql.connector
from mysql.connector import Error
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "database": os.getenv("DB_NAME", "csv_chatbot"),
    "user": os.getenv("DB_USER", ""),
    "password": os.getenv("DB_PASSWORD", "")
}

MYSQL_QUERY_CONFIG = {
    "host": os.getenv("MYSQL_QUERY_HOST", "127.0.0.1"),
    "port": int(os.getenv("MYSQL_QUERY_PORT", "3306")),
    "database": os.getenv("MYSQL_QUERY_NAME", ""),
    "user": os.getenv("MYSQL_QUERY_USER", "root"),
    "password": os.getenv("MYSQL_QUERY_PASSWORD", "Karna!21")
}

def get_db_connection(config: Dict[str, Any] = DB_CONFIG) -> Optional[mysql.connector.connection.MySQLConnection]:
    """Establish a database connection."""
    try:
        conn = mysql.connector.connect(
            host=config["host"],
            port=config["port"],
            database=config["database"],
            user=config["user"],
            password=config["password"],
            connection_timeout=10
        )
        logger.debug(f"Database connection established to {config['host']}")
        return conn
    except Error as e:
        logger.error(f"Database connection error: {e}")
        return None

def execute_query(connection: mysql.connector.connection.MySQLConnection, 
                 query: str, 
                 params: tuple = None, 
                 fetch: bool = True) -> Any:
    """Execute a SQL query."""
    cursor = None
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(query, params or ())
        if fetch:
            result = cursor.fetchall()
        else:
            result = cursor
        connection.commit()
        return result
    except Error as e:
        logger.error(f"Error executing query: {e}")
        connection.rollback()
        raise
    finally:
        if cursor:
            cursor.close()