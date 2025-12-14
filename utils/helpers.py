"""Helper functions for API retries, memory management, and utilities"""
import os
import gc
import time
import math
import json
import re
import base64
import threading
import streamlit as st
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import requests

# WebSocket keepalive configuration
WEBSOCKET_KEEPALIVE_INTERVAL = 5  # seconds between keepalive pings
WEBSOCKET_MAX_IDLE_TIME = 25  # max seconds before forcing a keepalive
_last_keepalive_time = time.time()


def _cleanup_session_state():
    """Clean up old/stale data from session state to prevent memory bloat."""
    MAX_CACHE_ENTRIES = 10
    MAX_SKILL_CACHE_SIZE = 500
    
    if 'jobs_cache' in st.session_state and isinstance(st.session_state.jobs_cache, dict):
        cache = st.session_state.jobs_cache
        if len(cache) > MAX_CACHE_ENTRIES:
            sorted_keys = sorted(
                cache.keys(),
                key=lambda k: cache[k].get('timestamp', ''),
                reverse=True
            )
            keys_to_remove = sorted_keys[MAX_CACHE_ENTRIES:]
            for key in keys_to_remove:
                del cache[key]
    
    if 'skill_embeddings_cache' in st.session_state:
        cache = st.session_state.skill_embeddings_cache
        if len(cache) > MAX_SKILL_CACHE_SIZE:
            keys_to_remove = list(cache.keys())[:-MAX_SKILL_CACHE_SIZE//2]
            for key in keys_to_remove:
                del cache[key]
    
    if 'user_skills_embeddings_cache' in st.session_state:
        cache = st.session_state.user_skills_embeddings_cache
        if len(cache) > MAX_SKILL_CACHE_SIZE:
            keys_to_remove = list(cache.keys())[:-MAX_SKILL_CACHE_SIZE//2]
            for key in keys_to_remove:
                del cache[key]
    
    gc.collect()


def get_img_as_base64(file):
    """Convert an image file to base64 string for embedding in HTML"""
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def _parse_retry_after_value(value):
    """Convert Retry-After style header values into seconds."""
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        seconds = float(value)
        if seconds >= 0:
            return int(math.ceil(seconds))
    except (ValueError, TypeError):
        pass
    if value.count(':') == 2:
        try:
            hours, minutes, seconds = value.split(':')
            total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(float(seconds))
            if total_seconds >= 0:
                return total_seconds
        except (ValueError, TypeError):
            pass
    try:
        retry_time = parsedate_to_datetime(value)
        if retry_time:
            if retry_time.tzinfo is None:
                retry_time = retry_time.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            delta = (retry_time - now).total_seconds()
            if delta > 0:
                return int(math.ceil(delta))
    except (TypeError, ValueError, OverflowError):
        pass
    return None


def _extract_delay_from_body(response):
    """Attempt to read retry hints from JSON/text error bodies."""
    if response is None:
        return None
    message = None
    try:
        data = response.json()
        if isinstance(data, dict):
            error = data.get('error') or {}
            if isinstance(error, dict):
                message = error.get('message') or error.get('code')
            if not message:
                message = data.get('message')
    except (ValueError, json.JSONDecodeError):
        pass
    if not message:
        message = response.text or ""
    if not message:
        return None
    match = re.search(r'after\s+(\d+)\s+seconds?', message, re.IGNORECASE)
    if match:
        try:
            seconds = int(match.group(1))
            if seconds >= 0:
                return seconds
        except ValueError:
            pass
    return None


def _determine_retry_delay(response, fallback_delay, max_delay):
    """Use headers/body hints to determine how long to wait before retrying."""
    if response is not None:
        headers = response.headers or {}
        header_candidates = [
            'Retry-After',
            'x-ms-retry-after-ms',
            'x-ms-retry-after',
            'x-ratelimit-reset-requests',
            'x-ratelimit-reset-tokens',
            'x-ratelimit-reset',
        ]
        for header in header_candidates:
            raw_value = headers.get(header)
            if not raw_value:
                continue
            if header.endswith('-ms'):
                try:
                    ms = float(raw_value)
                    if ms >= 0:
                        seconds = int(math.ceil(ms / 1000.0))
                        return max(1, min(seconds, max_delay)), f"header:{header}"
                except (ValueError, TypeError):
                    continue
            else:
                parsed = _parse_retry_after_value(raw_value)
                if parsed is not None:
                    return max(1, min(parsed, max_delay)), f"header:{header}"
        body_delay = _extract_delay_from_body(response)
        if body_delay is not None:
            return max(1, min(body_delay, max_delay)), "body"
    return max(1, min(fallback_delay, max_delay)), "fallback"


def _calculate_exponential_delay(initial_delay, attempt, max_delay):
    """Calculate exponential backoff delay for the current retry attempt."""
    return max(1, min(initial_delay * (2 ** attempt), max_delay))


def _chunked_sleep(delay, message_prefix=""):
    """Sleep in small chunks to prevent WebSocket timeout on Streamlit Cloud.
    
    This function breaks long sleeps into smaller chunks (max 2 seconds each)
    and sends UI updates to keep the WebSocket connection alive.
    """
    global _last_keepalive_time
    
    if delay <= 1:
        time.sleep(delay)
        _last_keepalive_time = time.time()
        return
    
    status_placeholder = st.empty()
    remaining = int(delay)
    while remaining > 0:
        if message_prefix:
            status_placeholder.caption(f"{message_prefix} ({remaining}s remaining...)")
        else:
            status_placeholder.caption(f"⏳ Processing... ({remaining}s)")
        # Use smaller chunks (1 second) for better responsiveness
        chunk = min(1, remaining)
        time.sleep(chunk)
        remaining -= chunk
        _last_keepalive_time = time.time()
    status_placeholder.empty()


def _websocket_keepalive(message=None, force=False):
    """Send a lightweight UI update to keep WebSocket connection alive.
    
    This function should be called during long-running operations to prevent
    the WebSocket connection from timing out. The function tracks the last
    keepalive time and only sends updates when necessary (unless force=True).
    
    Args:
        message: Optional status message to display
        force: If True, always send the keepalive regardless of timing
    """
    global _last_keepalive_time
    
    current_time = time.time()
    time_since_last = current_time - _last_keepalive_time
    
    # Only send keepalive if enough time has passed or force is True
    if not force and time_since_last < WEBSOCKET_KEEPALIVE_INTERVAL:
        return
    
    try:
        placeholder = st.empty()
        if message:
            placeholder.caption(f"⏳ {message}")
        else:
            # Send a minimal update to keep connection alive
            placeholder.empty()
        time.sleep(0.05)  # Brief pause to ensure message is sent
        placeholder.empty()
        _last_keepalive_time = time.time()
    except Exception:
        # Silently ignore errors - connection may already be closed
        pass


def _ensure_websocket_alive():
    """Check if we need to send a keepalive and do so if necessary.
    
    Call this function periodically during long operations to ensure
    the WebSocket connection stays alive.
    """
    global _last_keepalive_time
    
    current_time = time.time()
    time_since_last = current_time - _last_keepalive_time
    
    if time_since_last >= WEBSOCKET_MAX_IDLE_TIME:
        _websocket_keepalive(force=True)


class ProgressTracker:
    """Context manager for tracking progress of long-running operations.
    
    This class provides automatic WebSocket keepalive during long operations
    and optional progress bar updates.
    
    Example:
        with ProgressTracker("Processing jobs", total_steps=10) as tracker:
            for i in range(10):
                # Do work...
                tracker.update(i + 1, f"Step {i + 1}/10")
    """
    
    def __init__(self, description="Processing", total_steps=100, show_progress=True):
        self.description = description
        self.total_steps = total_steps
        self.show_progress = show_progress
        self.current_step = 0
        self.progress_bar = None
        self.status_text = None
        self._start_time = None
        self._last_update = 0
    
    def __enter__(self):
        self._start_time = time.time()
        self._last_update = time.time()
        if self.show_progress:
            self.progress_bar = st.progress(0, text=f"⏳ {self.description}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progress_bar:
            self.progress_bar.empty()
        if self.status_text:
            self.status_text.empty()
        return False
    
    def update(self, step=None, message=None):
        """Update progress and send keepalive if needed."""
        global _last_keepalive_time
        
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        progress = min(self.current_step / self.total_steps, 1.0)
        
        current_time = time.time()
        time_since_update = current_time - self._last_update
        
        # Update UI at least every WEBSOCKET_KEEPALIVE_INTERVAL seconds
        if time_since_update >= WEBSOCKET_KEEPALIVE_INTERVAL or step == self.total_steps:
            if self.show_progress and self.progress_bar:
                display_message = message or f"⏳ {self.description}... ({int(progress * 100)}%)"
                self.progress_bar.progress(progress, text=display_message)
            
            self._last_update = current_time
            _last_keepalive_time = current_time
    
    def set_message(self, message):
        """Update the status message without changing progress."""
        if self.show_progress and self.progress_bar:
            progress = self.current_step / self.total_steps
            self.progress_bar.progress(progress, text=f"⏳ {message}")
        _websocket_keepalive()


def api_call_with_retry(func, max_retries=3, initial_delay=1, max_delay=60):
    """Execute an API call with exponential backoff retry logic for rate limit errors (429)."""
    for attempt in range(max_retries):
        try:
            response = func()
            
            if response.status_code in [200, 201]:
                return response
            
            elif response.status_code == 429:
                if attempt < max_retries - 1:
                    fallback_delay = _calculate_exponential_delay(initial_delay, attempt, max_delay)
                    delay, delay_source = _determine_retry_delay(response, fallback_delay, max_delay)
                    source_note = ""
                    if delay_source != "fallback":
                        source_note = f" (server hint: {delay_source})"
                    if attempt == 0:
                        st.warning(
                            f"⏳ Rate limit reached. Retrying in {delay} seconds{source_note}... "
                            f"(Attempt {attempt + 1}/{max_retries})"
                        )
                    else:
                        st.caption(f"⏳ Retrying... ({attempt + 1}/{max_retries})")
                    _chunked_sleep(delay, f"⏳ Retry {attempt + 1}/{max_retries}")
                    continue
                else:
                    error_msg = (
                        "🚫 **Rate Limit Exceeded**\n\n"
                        "The API rate limit has been reached. Please:\n"
                        "1. Wait a few minutes and try again\n"
                        "2. Reduce the number of jobs you're searching for\n"
                        "3. Check your API quota/limits\n\n"
                        f"Status: {response.status_code}"
                    )
                    st.error(error_msg)
                    return None
            
            else:
                return response
                
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                delay = _calculate_exponential_delay(initial_delay, attempt, max_delay)
                st.warning(f"⏳ Request timed out. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                _chunked_sleep(delay)
                continue
            else:
                st.error("❌ Request timed out after multiple attempts. Please try again later.")
                return None
        
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                delay = _calculate_exponential_delay(initial_delay, attempt, max_delay)
                st.warning(f"⏳ Network error. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                _chunked_sleep(delay)
                continue
            else:
                st.error(f"❌ Network error after multiple attempts: {e}")
                return None
        
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            return None
    
    return None


def _is_streamlit_cloud():
    """Detect if running on Streamlit Cloud (ephemeral filesystem)."""
    return (
        os.environ.get('STREAMLIT_SHARING_MODE') is not None or
        os.environ.get('STREAMLIT_SERVER_PORT') is not None or
        os.path.exists('/mount/src') or
        'streamlit.app' in os.environ.get('HOSTNAME', '')
    )


def check_api_availability(show_messages=True):
    """Check which APIs are available based on configured credentials.
    
    Args:
        show_messages: Whether to show Streamlit messages about missing APIs
        
    Returns:
        Dictionary with availability status for each API service
    """
    # Import here to avoid circular import (config.py may import from utils)
    from config import Config
    
    # Initialize config if not already done
    if not Config._initialized:
        Config.setup()
    
    availability = {
        'azure_openai': False,
        'rapidapi': False,
        'pinecone': False,
    }
    
    missing_configs = []
    
    # Check Azure OpenAI
    if Config.AZURE_OPENAI_API_KEY and Config.AZURE_OPENAI_ENDPOINT:
        availability['azure_openai'] = True
    else:
        if not Config.AZURE_OPENAI_API_KEY:
            missing_configs.append('AZURE_OPENAI_API_KEY')
        if not Config.AZURE_OPENAI_ENDPOINT:
            missing_configs.append('AZURE_OPENAI_ENDPOINT')
    
    # Check RapidAPI
    if Config.RAPIDAPI_KEY:
        availability['rapidapi'] = True
    else:
        missing_configs.append('RAPIDAPI_KEY')
    
    # Check Pinecone
    if Config.PINECONE_API_KEY:
        availability['pinecone'] = True
    else:
        missing_configs.append('PINECONE_API_KEY')
    
    if show_messages and missing_configs:
        with st.expander("⚠️ Some API services are not configured", expanded=False):
            st.warning(f"Missing configuration: {', '.join(missing_configs)}")
            
            # Build configuration template based on what's actually missing
            config_lines = []
            if 'AZURE_OPENAI_API_KEY' in missing_configs:
                config_lines.append('AZURE_OPENAI_API_KEY = "your-azure-openai-key"')
            if 'AZURE_OPENAI_ENDPOINT' in missing_configs:
                config_lines.append('AZURE_OPENAI_ENDPOINT = "https://YOUR-RESOURCE.openai.azure.com"')
            if 'PINECONE_API_KEY' in missing_configs:
                config_lines.append('PINECONE_API_KEY = "your-pinecone-key"')
            if 'RAPIDAPI_KEY' in missing_configs:
                config_lines.append('RAPIDAPI_KEY = "your-rapidapi-key"')
            
            config_template = '\n'.join(config_lines)
            
            st.info(f"""
**To enable all features, configure the following in `.streamlit/secrets.toml`:**

```toml
{config_template}
```

**Features affected:**
- Without Azure OpenAI: Resume generation, AI analysis, profile extraction
- Without RapidAPI: Job search from LinkedIn and Indeed
- Without Pinecone: Vector-based semantic job matching
            """)
    
    return availability


def require_api(api_name: str, feature_name: str = "this feature") -> bool:
    """Check if a required API is available and show error if not.
    
    Args:
        api_name: Name of the API ('azure_openai', 'rapidapi', 'pinecone')
        feature_name: Human-readable name of the feature requiring this API
        
    Returns:
        True if API is available, False otherwise
    """
    availability = check_api_availability(show_messages=False)
    
    if not availability.get(api_name, False):
        if api_name == 'azure_openai':
            st.error(f"❌ Azure OpenAI API is required for {feature_name}")
            st.info("💡 Configure AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT in .streamlit/secrets.toml")
        elif api_name == 'rapidapi':
            st.error(f"❌ RapidAPI is required for {feature_name}")
            st.info("💡 Configure RAPIDAPI_KEY in .streamlit/secrets.toml to enable job search features")
        elif api_name == 'pinecone':
            st.error(f"❌ Pinecone API is required for {feature_name}")
            st.info("💡 Configure PINECONE_API_KEY in .streamlit/secrets.toml")
        return False
    
    return True
