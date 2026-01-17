"""
Database utilities for ChromaDB client management.
"""
import os
import chromadb
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_chromadb_client():
    """
    Get ChromaDB client based on environment.
    
    Returns:
        - CloudClient if CHROMA_CLOUD_API_KEY is set (production/deployment)
        - PersistentClient otherwise (development/local)
    
    Example:
        >>> client = get_chromadb_client()
        >>> collection = client.get_or_create_collection(name="recipes")
    """
    api_key = os.getenv("CHROMA_CLOUD_API_KEY")
    tenant = os.getenv("CHROMA_CLOUD_TENANT")
    database = os.getenv("CHROMA_CLOUD_DATABASE", "recipier")
    
    if api_key and tenant:
        # Production/Deployment: Use CloudClient
        return chromadb.CloudClient(
            api_key=api_key,
            tenant=tenant,
            database=database
        )
    else:
        # Development: Use PersistentClient
        db_path = os.getenv("CHROMA_DB_PATH", "./recipe_db")
        return chromadb.PersistentClient(path=db_path)
