"""
Indeed Job Scraper API Service.

This module provides the IndeedJobScraper class for searching jobs via
the Indeed Scraper API on RapidAPI.

Consolidated from modules/utils/api_clients.py for better organization.

Flow:
    ui/job_search_page.py
      ↓
    services/indeed_api.py
      → IndeedJobScraper.search_jobs()
      ↓
    modules/semantic_search/job_search.py
      → SemanticJobSearch.search()
      ↓
    core/job_matcher.py
      → JobMatcher.analyze_simple_match()
"""

import requests
import streamlit as st
from typing import List, Dict, Optional

# Import rate limiting from core
from core.rate_limiting import RateLimiter


class IndeedJobScraper:
    """Job scraper using Indeed Scraper API via RapidAPI.
    
    This class provides a clean interface for searching jobs from Indeed
    using the RapidAPI Indeed Scraper service.
    
    Attributes:
        api_key: RapidAPI API key
        url: API endpoint URL
        headers: Request headers including API key
        rate_limiter: Rate limiter for API calls
    """
    
    def __init__(self, api_key: str, max_requests_per_minute: int = 10):
        """Initialize the IndeedJobScraper.
        
        Args:
            api_key: RapidAPI API key for Indeed Scraper
            max_requests_per_minute: Rate limit for API calls
        """
        self.api_key = api_key
        self.url = "https://indeed-scraper-api.p.rapidapi.com/api/job"
        self.headers = {
            'Content-Type': 'application/json',
            'x-rapidapi-host': 'indeed-scraper-api.p.rapidapi.com',
            'x-rapidapi-key': api_key
        }
        
        # Import chunked sleep for WebSocket keepalive support
        try:
            from utils.helpers import _chunked_sleep
            sleep_func = _chunked_sleep
        except ImportError:
            import time
            sleep_func = lambda delay, msg="": time.sleep(delay)
        
        self.rate_limiter = RateLimiter(
            max_requests_per_minute=max_requests_per_minute,
            sleep_func=sleep_func
        )
    
    def search_jobs(
        self, 
        query: str, 
        location: str = "Hong Kong", 
        max_rows: int = 15, 
        job_type: str = "fulltime", 
        country: str = "hk",
        radius: str = "50",
        sort: str = "relevance",
        from_days: str = "7"
    ) -> List[Dict]:
        """Search for jobs using Indeed Scraper API.
        
        Includes WebSocket keepalive calls to prevent connection timeouts
        during the job search API call.
        
        Args:
            query: Job search query (e.g., "software engineer")
            location: Location to search in (e.g., "Hong Kong")
            max_rows: Maximum number of jobs to return
            job_type: Type of employment (fulltime, parttime, contract)
            country: Country code (hk, us, gb, ca, au, sg)
            radius: Search radius in miles
            sort: Sort order (relevance, date)
            from_days: Number of days to look back
            
        Returns:
            List of job dictionaries with parsed job data
        """
        # Import WebSocket utilities with fallback
        try:
            from utils.helpers import (
                _websocket_keepalive, 
                api_call_with_retry, 
                _ensure_websocket_alive
            )
        except ImportError:
            def _websocket_keepalive(message=None, force=False):
                pass
            def _ensure_websocket_alive():
                pass
            def api_call_with_retry(func, max_retries=3, initial_delay=1):
                try:
                    return func()
                except Exception:
                    return None
        
        payload = {
            "scraper": {
                "maxRows": max_rows,
                "query": query,
                "location": location,
                "jobType": job_type,
                "radius": radius,
                "sort": sort,
                "fromDays": from_days,
                "country": country
            }
        }
        
        try:
            _websocket_keepalive("Preparing job search...", force=True)
            self.rate_limiter.wait_if_needed()
            _websocket_keepalive("Searching jobs...")
            
            def make_request():
                return requests.post(
                    self.url, 
                    headers=self.headers, 
                    json=payload, 
                    timeout=60
                )
            
            response = api_call_with_retry(make_request, max_retries=3, initial_delay=3)
            
            # Keepalive after API response
            _ensure_websocket_alive()
            
            if response and response.status_code == 201:
                data = response.json()
                jobs = []
                
                _websocket_keepalive("Processing job results...")
                
                if 'returnvalue' in data and 'data' in data['returnvalue']:
                    job_list = data['returnvalue']['data']
                    
                    for idx, job_data in enumerate(job_list):
                        # Keepalive every 5 jobs during parsing
                        if idx % 5 == 0:
                            _ensure_websocket_alive()
                        parsed_job = self._parse_job(job_data)
                        if parsed_job:
                            jobs.append(parsed_job)
                
                _websocket_keepalive("Job search complete", force=True)
                return jobs
            else:
                if response:
                    if response.status_code == 404:
                        st.error("❌ Indeed API Error 404: Endpoint not found. The service may be temporarily unavailable.")
                        st.info("💡 Tip: Try using LinkedIn job search instead, or check if the API endpoint has changed.")
                    elif response.status_code == 401:
                        st.error("❌ Indeed API Error 401: Unauthorized. Please check your RapidAPI key.")
                        st.info("💡 Tip: Update your RAPIDAPI_KEY in .streamlit/secrets.toml")
                    elif response.status_code == 403:
                        st.error("❌ Indeed API Error 403: Forbidden. Your API key may not have access to this service.")
                    elif response.status_code == 429:
                        st.error("🚫 Rate limit reached for Indeed API. Please wait a few minutes and try again.")
                    else:
                        error_detail = response.text[:200] if response.text else "No error details"
                        st.error(f"❌ Indeed API Error {response.status_code}: {error_detail}")
                else:
                    st.error("❌ No response from Indeed API. Please check your network connection.")
                return []
                
        except Exception as e:
            st.error(f"Error searching jobs: {e}")
            return []
    
    def _parse_job(self, job_data: Dict) -> Optional[Dict]:
        """Parse job data from API response.
        
        Args:
            job_data: Raw job data from API response
            
        Returns:
            Parsed job dictionary or None if parsing fails
        """
        try:
            location_data = job_data.get('location', {})
            location = location_data.get('formattedAddressShort') or location_data.get('city', 'Hong Kong')
            
            job_types = job_data.get('jobType', [])
            job_type = ', '.join(job_types) if job_types else 'Full-time'
            
            benefits = job_data.get('benefits', [])
            attributes = job_data.get('attributes', [])
            
            full_description = job_data.get('descriptionText', 'No description')
            description = full_description[:50000] if len(full_description) > 50000 else full_description
            
            return {
                'title': job_data.get('title', 'N/A'),
                'company': job_data.get('companyName', 'N/A'),
                'location': location,
                'description': description,
                'salary': 'Not specified',
                'job_type': job_type,
                'url': job_data.get('jobUrl', '#'),
                'posted_date': job_data.get('age', 'Recently'),
                'benefits': benefits[:5],
                'skills': attributes[:10],
                'company_rating': job_data.get('rating', {}).get('rating', 0),
                'is_remote': job_data.get('isRemote', False)
            }
        except Exception:
            return None
    
    def test_connection(self) -> tuple:
        """Test API connection.
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        if not self.api_key:
            return False, "API key not configured"
        
        try:
            # Perform a minimal test search
            test_payload = {
                "scraper": {
                    "maxRows": 1,
                    "query": "test",
                    "location": "Hong Kong",
                    "country": "hk"
                }
            }
            
            response = requests.post(
                self.url, 
                headers=self.headers, 
                json=test_payload, 
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                return True, "Indeed API connection successful"
            elif response.status_code == 429:
                return False, "Rate limit exceeded - please wait before retrying"
            else:
                return False, f"API error: {response.status_code}"
                
        except requests.Timeout:
            return False, "Connection timeout"
        except Exception as e:
            return False, f"Connection error: {str(e)}"


# Backward compatibility alias
IndeedScraperAPI = IndeedJobScraper


def get_indeed_job_scraper() -> Optional[IndeedJobScraper]:
    """Get Indeed job scraper instance from session state.
    
    Returns:
        IndeedJobScraper instance or None if not configured
    """
    if 'indeed_scraper' not in st.session_state:
        rapidapi_key = st.secrets.get("RAPIDAPI_KEY", "")
        if not rapidapi_key:
            st.error("⚠️ RAPIDAPI_KEY is required in secrets.")
            return None
        st.session_state.indeed_scraper = IndeedJobScraper(rapidapi_key)
    return st.session_state.indeed_scraper
