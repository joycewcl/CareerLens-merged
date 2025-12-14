"""Validation functions"""
import streamlit as st


def validate_secrets():
    """Validate that required secrets are configured. Returns True if valid, False otherwise."""
    try:
        required_secrets = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "RAPIDAPI_KEY"]
        missing_secrets = []
        
        for secret in required_secrets:
            if not st.secrets.get(secret):
                missing_secrets.append(secret)
        
        if missing_secrets:
            st.error(f"""
            ⚠️ **Missing Required Configuration**
            
            The following secrets are not configured in your Streamlit app:
            - {', '.join(missing_secrets)}
            
            Please configure these in your Streamlit Cloud secrets or local `.streamlit/secrets.toml` file.
            See `.streamlit/secrets.toml.example` for the required format.
            """)
            return False
        
        return True
    except Exception as e:
        st.error(f"⚠️ Error validating secrets: {e}")
        return False
