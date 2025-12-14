"""
Utility modules for CareerLens application.

This package provides:
- Configuration constants and helpers
- API client classes (embedding, text generation, job scraping)
- Helper functions for retries, progress tracking, etc.
- Validation utilities
"""

from utils.config_utils import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_MAX_JOBS_TO_INDEX,
    EMBEDDING_BATCH_DELAY,
    RAPIDAPI_MAX_REQUESTS_PER_MINUTE,
    ENABLE_PROFILE_PASS2,
    USE_FAST_SKILL_MATCHING,
    _determine_index_limit
)
from utils.helpers import (
    _cleanup_session_state,
    get_img_as_base64,
    api_call_with_retry,
    _websocket_keepalive,
    _chunked_sleep,
    _is_streamlit_cloud,
    _ensure_websocket_alive,
    ProgressTracker
)
from utils.api_clients import (
    APIMEmbeddingGenerator,
    AzureOpenAITextGenerator,
    RateLimiter,
    IndeedScraperAPI,
    TokenUsageTracker,
    get_token_tracker,
    get_embedding_generator,
    get_text_generator,
    get_job_scraper
)
from utils.validation import validate_secrets

__all__ = [
    # Config
    'DEFAULT_EMBEDDING_BATCH_SIZE',
    'DEFAULT_MAX_JOBS_TO_INDEX',
    'EMBEDDING_BATCH_DELAY',
    'RAPIDAPI_MAX_REQUESTS_PER_MINUTE',
    'ENABLE_PROFILE_PASS2',
    'USE_FAST_SKILL_MATCHING',
    '_determine_index_limit',
    
    # Helpers
    '_cleanup_session_state',
    'get_img_as_base64',
    'api_call_with_retry',
    '_websocket_keepalive',
    '_chunked_sleep',
    '_is_streamlit_cloud',
    '_ensure_websocket_alive',
    'ProgressTracker',
    
    # API Clients
    'APIMEmbeddingGenerator',
    'AzureOpenAITextGenerator',
    'RateLimiter',
    'IndeedScraperAPI',
    'TokenUsageTracker',
    'get_token_tracker',
    'get_embedding_generator',
    'get_text_generator',
    'get_job_scraper',
    
    # Validation
    'validate_secrets',
]
