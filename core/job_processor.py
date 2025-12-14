"""
Job processing orchestration classes.

This module contains the main orchestrator classes for:
- Resume processing and AI analysis
- Job searching via external APIs
- Semantic matching and scoring
"""

import time
from typing import Dict, List, Tuple

from config import Config
from core.resume_parser import ResumeParser, GPT4JobRoleDetector
from core.job_matcher import JobMatcher, calculate_match_scores, calculate_job_match_score
from services.linkedin_api import LinkedInJobSearcher


class JobSeekerBackend:
    """Main backend with FULL integration - optimized for fast startup.
    
    This class orchestrates the entire job matching workflow:
    - Resume parsing and AI analysis
    - Job searching via LinkedIn/Indeed APIs
    - Semantic matching using Pinecone
    - Skill-based scoring
    """
    
    def __init__(self):
        print("🚀 Initializing Job Matcher Backend (lightweight)...")
        Config.validate()
        
        # Lightweight components - instant init
        self.resume_parser = ResumeParser()
        self.gpt4_detector = GPT4JobRoleDetector()
        
        # Lazy-load heavy components - deferred until first use
        self._job_searcher = None
        self._matcher = None
        
        print("✅ Backend initialized (fast mode)!\n")
    
    @property
    def matcher(self):
        """Lazy-load JobMatcher only when needed."""
        if self._matcher is None:
            print("📦 Loading JobMatcher (first use)...")
            self._matcher = JobMatcher()
        return self._matcher
    
    @property
    def job_searcher(self):
        """Lazy initialization of job searcher - only tests connection when first used."""
        if self._job_searcher is None:
            print("\n🧪 Initializing RapidAPI job searcher...")
            
            # Check if RAPIDAPI_KEY is configured
            if not Config.RAPIDAPI_KEY:
                print("⚠️ WARNING: RAPIDAPI_KEY is not configured!")
                print("   Job search functionality will not work.")
                print("   Please configure RAPIDAPI_KEY in your Streamlit secrets.")
                self._job_searcher = LinkedInJobSearcher("")
                return self._job_searcher
            
            self._job_searcher = LinkedInJobSearcher(Config.RAPIDAPI_KEY)
            # Test API connection only once
            is_working, message = self._job_searcher.test_api_connection()
            if is_working:
                print(f"✅ {message}")
            else:
                print(f"⚠️ WARNING: {message}")
                print("   Job search may not work properly!")
        return self._job_searcher
    
    def test_api_connection(self):
        """Test API connection on demand (not at startup)."""
        return self.job_searcher.test_api_connection()
    
    def process_resume(self, file_obj, filename: str) -> Tuple[Dict, Dict]:
        """Process resume and get AI analysis.
        
        Args:
            file_obj: File-like object containing the resume
            filename: Original filename with extension
            
        Returns:
            Tuple of (resume_data, ai_analysis)
        """
        print(f"📄 Processing resume: {filename}")
        
        # Parse resume
        resume_data = self.resume_parser.parse_resume(file_obj, filename)
        print(f"✅ Extracted {resume_data['word_count']} words from resume")
        
        # Get GPT-4 analysis
        ai_analysis = self.gpt4_detector.analyze_resume_for_job_roles(resume_data)
        
        # Add skills to resume_data
        resume_data['skills'] = ai_analysis.get('skills', [])
        
        return resume_data, ai_analysis
    
    def search_and_match_jobs(self, resume_data: Dict, ai_analysis: Dict, num_jobs: int = 30, 
                               search_keywords: str = None, location: str = None) -> List[Dict]:
        """Search for jobs and rank by match quality.
        
        Args:
            resume_data: Parsed resume data
            ai_analysis: AI-extracted skills and role analysis
            num_jobs: Number of jobs to search for
            search_keywords: Search keywords (if None, uses ai_analysis primary_role)
            location: Location preference (if None, defaults to Hong Kong)
            
        Returns:
            List of matched jobs sorted by combined score
        """
        # Use provided keywords or fall back to AI-detected role
        primary_role = ai_analysis.get('primary_role', '')
        search_query = search_keywords if search_keywords else primary_role
        
        # If no search query available, return empty
        if not search_query or not search_query.strip():
            print("⚠️ No search keywords provided and no primary role detected.")
            print("   Please provide search keywords in your profile.")
            return []
        
        location = location if location else "Hong Kong"
        
        print(f"\n{'='*60}")
        print(f"🔍 SEARCHING JOBS")
        print(f"{'='*60}")
        print(f"🔍 Search Query: {search_query}")
        print(f"📍 Location: {location}")
        print(f"{'='*60}\n")
        
        # Search jobs
        jobs = self.job_searcher.search_jobs(
            keywords=search_query,
            location=location,
            limit=num_jobs
        )
        
        if not jobs or len(jobs) == 0:
            print("\n❌ No jobs found from RapidAPI")
            print("\n💡 Possible reasons:")
            print("   - API key might be invalid/expired")
            print("   - Rate limit exceeded")
            print("   - No jobs available for this search term")
            return []
        
        print(f"\n✅ Retrieved {len(jobs)} jobs from RapidAPI")
        print(f"📊 Indexing jobs in Pinecone...")
        
        # Index jobs
        indexed = self.matcher.index_jobs(jobs)
        print(f"✅ Indexed {indexed} jobs in vector database")
        
        # Wait for indexing
        print("⏳ Waiting for indexing to complete...")
        time.sleep(1)
        
        # Match resume to jobs
        print(f"\n🎯 MATCHING & RANKING JOBS")
        print(f"{'='*60}")
        matched_jobs = self.matcher.search_similar_jobs(
            resume_data, 
            ai_analysis, 
            top_k=min(20, len(jobs))
        )
        
        if not matched_jobs:
            print("⚠️ No matches found")
            return []
        
        # Calculate match scores
        matched_jobs = calculate_match_scores(matched_jobs, ai_analysis)
        
        # Sort by combined score
        matched_jobs.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        
        print(f"✅ Ranked {len(matched_jobs)} jobs by match quality")
        print(f"{'='*60}\n")
        
        return matched_jobs
    
    @staticmethod
    def parse_cv_with_ai(cv_text):
        """Parse CV text with AI to extract structured fields.
        
        Args:
            cv_text: Raw CV text content
            
        Returns:
            Dictionary with extracted fields
        """
        from openai import AzureOpenAI
        import json
        import httpx
        
        # Clean endpoint to prevent double /openai path issues
        endpoint = Config.AZURE_OPENAI_ENDPOINT
        if endpoint:
            endpoint = endpoint.rstrip('/')
            if endpoint.endswith('/openai'):
                endpoint = endpoint[:-7]
        
        # Create a custom http client that ignores SSL errors
        http_client = httpx.Client(verify=False)

        # Use Azure OpenAI client
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=Config.AZURE_OPENAI_API_KEY,
            api_version=Config.AZURE_OPENAI_API_VERSION,
            http_client=http_client
        )
        
        prompt = f"""
Below is the cv text of a candidate. 
Please extract structured information (leave blank if missing):
cv_text: '''{cv_text}'''

Please output JSON, fields including:
- education_level(doctor/master/bachelor/associate/highschool)
- major
- graduation_status(fresh graduate/experienced/in study)
- university_background(985 university/211 university/overseas university/regular university/other)
- languages
- certificates
- hard_skills
- soft_skills
- work_experience(fresh graduate/1-3 years/3-5 years/5-10 years/10+ years)
- project_experience
- location_preference
- industry_preference
- salary_expectation
- benefits_expectation

Please return the result in the JSON format only, no extra explanation.
"""

        response = client.chat.completions.create(
            model=Config.AZURE_OPENAI_DEPLOYMENT or "gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        try:
            return json.loads(response.choices[0].message.content)
        except Exception:
            return {}


class JobMatcherBackend:
    """Backend implementation using JSearch API for job fetching."""
    
    def fetch_real_jobs(self, search_query, location="", country="us", num_pages=1):
        """Get actual job data from JSearch API.
        
        Args:
            search_query: Job search query
            location: Location filter
            country: Country code
            num_pages: Number of result pages to fetch
            
        Returns:
            List of job dictionaries
        """
        import requests
        
        try:
            # JSearch API configuration
            API_KEY = Config.RAPIDAPI_KEY or "your_jsearch_api_key_here"
            BASE_URL = "https://jsearch.p.rapidapi.com/search"
            
            headers = {
                "X-RapidAPI-Key": API_KEY,
                "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
            }
            
            all_jobs = []
            
            for page in range(1, num_pages + 1):
                querystring = {
                    "query": f"{search_query} {location}",
                    "page": str(page),
                    "num_pages": "1"
                }
                
                response = requests.get(BASE_URL, headers=headers, params=querystring)
                
                if response.status_code == 200:
                    data = response.json()
                    jobs = data.get('data', [])
                    all_jobs.extend(jobs)
                    print(f"✅ Page {page} fetched {len(jobs)} jobs")
                else:
                    print(f"❌ API request failed: {response.status_code}")
                    break
            
            print(f"🎯 Found total of {len(all_jobs)} positions")
            return all_jobs
            
        except Exception as e:
            print(f"❌ Failed to fetch jobs: {e}")
            return self.get_mock_jobs(search_query, location)

    def get_mock_jobs(self, search_query, location):
        """Return mock job data (used when API is unavailable)."""
        print("🔄 Using simulated data...")
        
        mock_jobs = [
            {
                'job_title': f'Senior {search_query}',
                'employer_name': 'Tech Company Inc.',
                'job_city': location or 'Hong Kong',
                'job_country': 'HK',
                'job_employment_type': 'FULLTIME',
                'job_posted_at': '2024-01-15',
                'job_description': f'We are looking for a skilled {search_query} to join our team.',
                'job_apply_link': 'https://example.com/apply/1',
                'job_highlights': {
                    'Qualifications': ['Bachelor\'s degree', '3+ years of experience'],
                    'Responsibilities': ['Develop applications', 'Collaborate with team']
                }
            },
            {
                'job_title': f'Junior {search_query}',
                'employer_name': 'Startup Solutions',
                'job_city': location or 'Hong Kong',
                'job_country': 'HK',
                'job_employment_type': 'FULLTIME',
                'job_posted_at': '2024-01-10',
                'job_description': f'Entry-level position for {search_query}.',
                'job_apply_link': 'https://example.com/apply/2',
                'job_highlights': {
                    'Qualifications': ['Degree in related field'],
                    'Responsibilities': ['Assist senior developers']
                }
            },
        ]
        
        return mock_jobs

    def calculate_job_match_score(self, job_seeker_data, job_data):
        """Calculate job match score between job seeker and job data."""
        return calculate_job_match_score(job_seeker_data, job_data)
