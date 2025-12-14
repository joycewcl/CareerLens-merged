"""
LinkedIn Job Search API client.

This module provides the LinkedInJobSearcher class for searching jobs
using the RapidAPI LinkedIn Job Search API.
"""

import requests
from typing import Dict, List, Tuple


class LinkedInJobSearcher:
    """Search for jobs using RapidAPI LinkedIn API.
    
    This class handles job searches via RapidAPI's LinkedIn Job Search endpoint.
    It includes query simplification, fallback searches, and job normalization.
    """
    
    def __init__(self, api_key: str):
        """Initialize the LinkedIn job searcher.
        
        Args:
            api_key: RapidAPI key for authentication
        """
        self.api_key = api_key
        self.base_url = "https://linkedin-job-search-api.p.rapidapi.com/active-jb-7d"
        self.headers = {
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": "linkedin-job-search-api.p.rapidapi.com"
        }
    
    def test_api_connection(self) -> Tuple[bool, str]:
        """Test if the API is working.
        
        Returns:
            Tuple of (success_status, message)
        """
        try:
            querystring = {
                "limit": "5",
                "offset": "0",
                "title_filter": "\"Engineer\"",
                "location_filter": "\"Hong Kong\"",
                "description_type": "text"
            }
            
            response = requests.get(
                self.base_url,
                headers=self.headers,
                params=querystring,
                timeout=10
            )
            
            if response.status_code == 200:
                return True, "API is working"
            elif response.status_code == 403:
                return False, "API key is invalid or expired (403 Forbidden)"
            elif response.status_code == 429:
                return False, "Rate limit exceeded (429 Too Many Requests)"
            else:
                return False, f"API returned status code {response.status_code}"
        
        except Exception as e:
            return False, f"Connection error: {str(e)}"
    
    def search_jobs(
        self,
        keywords: str,
        location: str = "Hong Kong",
        limit: int = 20
    ) -> List[Dict]:
        """Search LinkedIn jobs with simplified queries.
        
        Args:
            keywords: Search keywords/job title
            location: Location to search in
            limit: Maximum number of jobs to return
            
        Returns:
            List of normalized job dictionaries
        """
        # Simplify complex queries
        simple_keywords = self._simplify_query(keywords)
        
        querystring = {
            "limit": str(limit),
            "offset": "0",
            "title_filter": f'"{simple_keywords}"',
            "location_filter": f'"{location}"',
            "description_type": "text"
        }
        
        try:
            print(f"🔍 Searching RapidAPI...")
            print(f"   Original query: {keywords}")
            print(f"   Simplified to: {simple_keywords}")
            print(f"   Location: {location}")
            
            response = requests.get(
                self.base_url, 
                headers=self.headers, 
                params=querystring, 
                timeout=30
            )
            
            print(f"📊 API Response Status: {response.status_code}")
            
            if response.status_code == 404:
                print("❌ API Error: 404 Not Found")
                print("   The API endpoint may have changed or is unavailable")
                print("   Please check the LinkedIn Job Search API documentation")
                st.error("❌ LinkedIn Job API endpoint not found (404). The service may be temporarily unavailable.")
                return []
            
            elif response.status_code == 403:
                print("❌ API Key Error: 403 Forbidden")
                print("   Your RapidAPI key might be invalid or expired")
                print("   Check: https://rapidapi.com/")
                st.error("❌ Invalid RapidAPI key. Please update your RAPIDAPI_KEY in .streamlit/secrets.toml")
                return []
            
            elif response.status_code == 429:
                print("❌ Rate Limit: 429 Too Many Requests")
                print("   Wait a few minutes or upgrade your RapidAPI plan")
                st.warning("⚠️ Rate limit exceeded. Please wait a few minutes before searching again.")
                return []
            
            elif response.status_code != 200:
                print(f"❌ API Error: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                st.error(f"❌ LinkedIn Job API Error {response.status_code}: {response.text[:200]}")
                return []
            
            data = response.json()
            
            # Handle different response formats
            if isinstance(data, list):
                jobs = data
            elif isinstance(data, dict):
                jobs = data.get('data', data.get('jobs', data.get('results', [])))
            else:
                jobs = []
            
            if not jobs:
                print(f"⚠️ No jobs found for '{simple_keywords}'")
                print("   Trying fallback searches...")
                
                # Try alternative searches
                for alternative in self._get_alternative_searches(simple_keywords):
                    alt_jobs = self._try_alternative_search(alternative, location, 10)
                    if alt_jobs:
                        print(f"✅ Found {len(alt_jobs)} jobs with alternative search: {alternative}")
                        jobs.extend(alt_jobs)
                        if len(jobs) >= 10:
                            break
            
            normalized = self._normalize_jobs(jobs)
            print(f"✅ Retrieved {len(normalized)} jobs from RapidAPI")
            return normalized
            
        except Exception as e:
            print(f"❌ LinkedIn API Error: {str(e)}")
            return []
    
    def _simplify_query(self, query: str) -> str:
        """Simplify complex boolean queries to simple terms.
        
        Args:
            query: Original search query (may contain boolean operators)
            
        Returns:
            Simplified query string
        """
        # Remove boolean operators and parentheses
        simple = query.replace(" OR ", " ").replace(" AND ", " ")
        simple = simple.replace("(", "").replace(")", "")
        simple = simple.replace('"', "")
        
        # Take first few words (most important)
        words = simple.split()[:3]
        return " ".join(words)
    
    def _get_alternative_searches(self, primary_query: str) -> List[str]:
        """Generate alternative search terms.
        
        Args:
            primary_query: The primary search query that returned no results
            
        Returns:
            List of alternative search terms to try
        """
        alternatives = [
            primary_query.split()[0] if primary_query.split() else primary_query,  # First word only
            "Manager",  # Generic fallback
            "Analyst",  # Generic fallback
        ]
        return alternatives
    
    def _try_alternative_search(self, keywords: str, location: str, limit: int) -> List[Dict]:
        """Try an alternative search.
        
        Args:
            keywords: Alternative search keywords
            location: Location to search in
            limit: Maximum number of results
            
        Returns:
            List of raw job dictionaries from API
        """
        try:
            querystring = {
                "limit": str(limit),
                "offset": "0",
                "title_filter": f'"{keywords}"',
                "location_filter": f'"{location}"',
                "description_type": "text"
            }
            
            response = requests.get(
                self.base_url,
                headers=self.headers,
                params=querystring,
                timeout=20
            )
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return data.get('data', data.get('jobs', data.get('results', [])))
            
            return []
        
        except Exception:
            return []
    
    def _normalize_jobs(self, jobs: List[Dict]) -> List[Dict]:
        """Normalize job structure from API response.
        
        Args:
            jobs: Raw job dictionaries from API
            
        Returns:
            List of normalized job dictionaries
        """
        normalized_jobs = []
        
        for job in jobs:
            try:
                # Handle location
                location = "Remote"
                if job.get('locations_derived') and len(job['locations_derived']) > 0:
                    location = job['locations_derived'][0]
                elif job.get('locations_raw'):
                    try:
                        loc_raw = job['locations_raw'][0]
                        if isinstance(loc_raw, dict) and 'address' in loc_raw:
                            addr = loc_raw['address']
                            city = addr.get('addressLocality', '')
                            region = addr.get('addressRegion', '')
                            if city and region:
                                location = f"{city}, {region}"
                    except (KeyError, TypeError, IndexError):
                        pass
                
                normalized_job = {
                    'id': job.get('id', f"job_{len(normalized_jobs)}"),
                    'title': job.get('title', 'Unknown Title'),
                    'company': job.get('organization', 'Unknown Company'),
                    'location': location,
                    'description': job.get('description_text', ''),
                    'url': job.get('url', ''),
                    'posted_date': job.get('date_posted', 'Unknown'),
                }
                
                normalized_jobs.append(normalized_job)
                
            except Exception:
                continue
        
        return normalized_jobs


# Cached instance management
_cached_linkedin_searcher = None


def get_linkedin_job_searcher(api_key: str = None) -> LinkedInJobSearcher:
    """Get cached LinkedInJobSearcher - only initializes once.
    
    Args:
        api_key: RapidAPI key. If not provided, will try to get from Config.
        
    Returns:
        LinkedInJobSearcher instance
    """
    global _cached_linkedin_searcher
    
    if _cached_linkedin_searcher is None:
        if api_key is None:
            from config import Config
            api_key = Config.RAPIDAPI_KEY
        
        print("📦 Initializing LinkedIn Job Searcher (first time only)...")
        _cached_linkedin_searcher = LinkedInJobSearcher(api_key)
        print("✅ LinkedIn Job Searcher cached!")
    
    return _cached_linkedin_searcher
