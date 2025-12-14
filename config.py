# config.py
"""
Configuration module for CareerLens application.

This module provides centralized configuration management, including:
- Database paths
- API keys and endpoints
- Model configuration
- Validation utilities
"""
import os
from pathlib import Path
from typing import Tuple

# Try to import streamlit for secrets access
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


def _get_secret(key: str, default=None):
    """Get a configuration value from environment or Streamlit secrets.
    
    Args:
        key: Configuration key name
        default: Default value if not found
        
    Returns:
        Configuration value or default
    """
    # Try environment variable first
    value = os.getenv(key)
    if value:
        return value
    
    # Try Streamlit secrets
    if STREAMLIT_AVAILABLE:
        try:
            value = st.secrets.get(key)
            if value:
                return value
        except Exception:
            pass
    
    return default


class Config:
    """Central configuration class for CareerLens application."""
    
    # Project root
    PROJECT_ROOT = Path(__file__).parent.absolute()
    
    # Database paths
    DB_PATH_JOB_SEEKER = os.getenv(
        'DB_PATH_JOB_SEEKER',
        str(PROJECT_ROOT / 'job_seeker.db')
    )
    DB_PATH_HEAD_HUNTER = os.getenv(
        'DB_PATH_HEAD_HUNTER',
        str(PROJECT_ROOT / 'head_hunter_jobs.db')
    )
    DB_PATH_CHROMA = os.getenv(
        'DB_PATH_CHROMA',
        str(PROJECT_ROOT / '.chroma_db')
    )
    
    # Azure OpenAI Configuration
    AZURE_OPENAI_API_KEY = None
    AZURE_OPENAI_ENDPOINT = None
    AZURE_OPENAI_API_VERSION = '2024-02-15-preview'
    AZURE_OPENAI_DEPLOYMENT = None
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = None
    
    # Aliases for backward compatibility
    AZURE_API_KEY = None
    AZURE_ENDPOINT = None
    AZURE_API_VERSION = '2024-02-15-preview'
    AZURE_MODEL = 'gpt-4o-mini'
    
    # RapidAPI Configuration
    RAPIDAPI_KEY = None
    
    # Pinecone Configuration
    PINECONE_API_KEY = None
    PINECONE_ENVIRONMENT = 'us-east-1'
    
    # Model Configuration
    MODEL_NAME = 'all-MiniLM-L6-v2'
    INDEX_NAME = 'job-matcher'
    EMBEDDING_DIMENSION = 384
    
    _initialized = False
    
    @classmethod
    def setup(cls):
        """Initialize configuration from environment and secrets.
        
        This method should be called once at application startup.
        """
        if cls._initialized:
            return
        
        # Azure OpenAI
        cls.AZURE_OPENAI_API_KEY = _get_secret('AZURE_OPENAI_API_KEY')
        cls.AZURE_OPENAI_ENDPOINT = _get_secret('AZURE_OPENAI_ENDPOINT')
        
        # Normalize Azure OpenAI endpoint - remove trailing /openai if present
        # Azure OpenAI SDK expects base endpoint without the /openai path
        if cls.AZURE_OPENAI_ENDPOINT:
            cls.AZURE_OPENAI_ENDPOINT = cls.AZURE_OPENAI_ENDPOINT.rstrip('/')
            if cls.AZURE_OPENAI_ENDPOINT.endswith('/openai'):
                cls.AZURE_OPENAI_ENDPOINT = cls.AZURE_OPENAI_ENDPOINT[:-7]
        
        cls.AZURE_OPENAI_API_VERSION = _get_secret('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
        cls.AZURE_OPENAI_DEPLOYMENT = _get_secret('AZURE_OPENAI_DEPLOYMENT', 'gpt-4o-mini')
        cls.AZURE_OPENAI_EMBEDDING_DEPLOYMENT = _get_secret('AZURE_OPENAI_EMBEDDING_DEPLOYMENT', 'text-embedding-3-small')
        
        # Aliases for backward compatibility
        cls.AZURE_API_KEY = cls.AZURE_OPENAI_API_KEY
        cls.AZURE_ENDPOINT = cls.AZURE_OPENAI_ENDPOINT
        cls.AZURE_API_VERSION = cls.AZURE_OPENAI_API_VERSION
        cls.AZURE_MODEL = cls.AZURE_OPENAI_DEPLOYMENT or 'gpt-4o-mini'
        
        # RapidAPI
        cls.RAPIDAPI_KEY = _get_secret('RAPIDAPI_KEY')
        
        # Pinecone
        cls.PINECONE_API_KEY = _get_secret('PINECONE_API_KEY')
        cls.PINECONE_ENVIRONMENT = _get_secret('PINECONE_ENVIRONMENT', 'us-east-1')
        
        # Model Configuration
        cls.MODEL_NAME = _get_secret('MODEL_NAME', 'all-MiniLM-L6-v2')
        cls.INDEX_NAME = _get_secret('INDEX_NAME', 'job-matcher')
        cls.EMBEDDING_DIMENSION = int(_get_secret('EMBEDDING_DIMENSION', '384'))
        
        cls._initialized = True
    
    @classmethod
    def validate(cls):
        """Validate that required configuration is available.
        
        This method prints warnings for missing configuration but does not raise errors.
        """
        if not cls._initialized:
            cls.setup()
        
        missing = []
        
        if not cls.AZURE_OPENAI_API_KEY:
            missing.append('AZURE_OPENAI_API_KEY')
        if not cls.AZURE_OPENAI_ENDPOINT:
            missing.append('AZURE_OPENAI_ENDPOINT')
        if not cls.RAPIDAPI_KEY:
            missing.append('RAPIDAPI_KEY')
        if not cls.PINECONE_API_KEY:
            missing.append('PINECONE_API_KEY')
        
        if missing:
            print(f"⚠️ Missing configuration: {', '.join(missing)}")
            print("   Some features may not work properly.")
    
    @classmethod
    def check_azure_credentials(cls) -> Tuple[bool, str]:
        """Check if Azure OpenAI credentials are configured.
        
        Returns:
            Tuple of (is_configured, error_message)
        """
        if not cls._initialized:
            cls.setup()
        
        if not cls.AZURE_OPENAI_API_KEY:
            return False, "AZURE_OPENAI_API_KEY is not configured"
        if not cls.AZURE_OPENAI_ENDPOINT:
            return False, "AZURE_OPENAI_ENDPOINT is not configured"
        
        return True, ""
    
    @classmethod
    def get_azure_client_config(cls) -> dict:
        """Get configuration dictionary for Azure OpenAI client.
        
        Returns:
            Dictionary with api_key, azure_endpoint, and api_version
        """
        if not cls._initialized:
            cls.setup()
        
        return {
            'api_key': cls.AZURE_OPENAI_API_KEY,
            'azure_endpoint': cls.AZURE_OPENAI_ENDPOINT,
            'api_version': cls.AZURE_OPENAI_API_VERSION,
        }


# Auto-initialize configuration on module import
Config.setup()
