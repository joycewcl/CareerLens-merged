"""
Configuration for Job Matcher
Reads sensitive values from Streamlit secrets (.streamlit/secrets.toml)
"""
import os
import streamlit as st


class _ConfigMeta(type):
    """Metaclass to allow class-level property access for backward compatibility.
    
    This allows code like `Config.AZURE_ENDPOINT` to dynamically read from secrets,
    rather than returning a static value.
    """
    
    @property
    def PINECONE_API_KEY(cls):
        return cls.get_pinecone_api_key()
    
    @property
    def AZURE_ENDPOINT(cls):
        return cls.get_azure_endpoint()
    
    @property
    def AZURE_API_KEY(cls):
        return cls.get_azure_api_key()
    
    @property
    def RAPIDAPI_KEY(cls):
        return cls.get_rapidapi_key()


class Config(metaclass=_ConfigMeta):
    """Configuration settings - reads sensitive values from Streamlit secrets.
    
    API keys and endpoints are read from .streamlit/secrets.toml (local)
    or Streamlit Cloud secrets (production).
    
    Usage:
        Config.AZURE_ENDPOINT  # Returns endpoint from secrets
        Config.AZURE_API_KEY   # Returns API key from secrets
    """
    
    # Pinecone (non-sensitive settings)
    PINECONE_ENVIRONMENT = "us-east-1"
    INDEX_NAME = "job-resume-matcher"
    EMBEDDING_DIMENSION = 384
    MODEL_NAME = "all-MiniLM-L6-v2"
    
    # Azure OpenAI (non-sensitive settings)
    AZURE_API_VERSION = "2024-10-21"
    AZURE_MODEL = "gpt-4o-mini"
    
    # Application Settings
    MAX_JOBS_TO_FETCH = 50
    TOP_MATCHES_TO_SHOW = 5
    UPLOAD_FOLDER = "uploads"
    
    @staticmethod
    def get_pinecone_api_key():
        """Get Pinecone API key from Streamlit secrets."""
        return st.secrets.get("PINECONE_API_KEY", "")
    
    @staticmethod
    def get_azure_endpoint():
        """Get Azure OpenAI endpoint from secrets.
        
        Users should set: AZURE_OPENAI_ENDPOINT = "https://hkust.azure-api.net/openai"
        This method automatically strips the /openai suffix because the SDK adds it internally.
        """
        endpoint = st.secrets.get("AZURE_OPENAI_ENDPOINT", "")
        # Strip /openai suffix if present (SDK adds it automatically)
        if endpoint.endswith('/openai'):
            endpoint = endpoint[:-7]
        return endpoint.rstrip('/')
    
    @staticmethod
    def get_azure_api_key():
        """Get Azure OpenAI API key from Streamlit secrets."""
        return st.secrets.get("AZURE_OPENAI_API_KEY", "")
    
    @staticmethod
    def get_rapidapi_key():
        """Get RapidAPI key from Streamlit secrets."""
        return st.secrets.get("RAPIDAPI_KEY", "")
    
    @classmethod
    def setup(cls):
        """Create necessary directories."""
        os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)
    
    @classmethod
    def validate(cls):
        """Validate that required API keys are set in secrets."""
        missing = []
        if not cls.get_azure_api_key():
            missing.append("AZURE_OPENAI_API_KEY")
        if not cls.get_azure_endpoint():
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not cls.get_rapidapi_key():
            missing.append("RAPIDAPI_KEY")
        
        if missing:
            print(f"⚠️ Missing secrets: {', '.join(missing)}")
            return False
        
        print("✅ Configuration validated")
        return True
