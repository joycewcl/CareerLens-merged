"""
Job Matcher Backend - REFACTORED VERSION

This file serves as a facade that re-exports components from their new modular locations.
All components have been reorganized into appropriate modules:

- core/job_matcher.py: JobMatcher class and match scoring functions
- core/resume_parser.py: ResumeParser, GPT4JobRoleDetector, profile extraction
- core/interview.py: AI interview functions
- core/rate_limiting.py: TokenUsageTracker, RateLimiter
- services/linkedin_api.py: LinkedInJobSearcher
- services/azure_openai.py: Resume formatters, Azure OpenAI clients
- modules/analysis/match_analysis.py: Salary/domain filtering
- database/queries.py: Database query functions
"""

import time
from typing import Dict, List, Tuple, Optional
import streamlit as st

# Initialize config first
from config import Config
Config.setup()

# ============================================================================
# IMPORTS FROM REORGANIZED MODULES
# ============================================================================

# Core rate limiting (canonical source)
from core.rate_limiting import TokenUsageTracker, RateLimiter

# Job matcher and scoring
from core.job_matcher import (
    JobMatcher,
    calculate_match_scores,
    analyze_match_simple,
    calculate_job_match_score,
    _get_sentence_transformer_model_cached,
    _get_pinecone_client_cached,
    _get_pinecone_index_cached,
)

# Resume parsing and profile extraction
from core.resume_parser import (
    ResumeParser,
    GPT4JobRoleDetector,
    extract_relevant_resume_sections,
    extract_structured_profile,
    generate_tailored_resume,
)

# AI Interview functions
from core.interview import (
    initialize_interview_session,
    generate_interview_question,
    evaluate_answer,
    generate_final_summary,
    ai_interview_page,
)

# LinkedIn job search
from services.linkedin_api import (
    LinkedInJobSearcher,
    get_linkedin_job_searcher,
)

# Resume formatters and Azure OpenAI
from services.azure_openai import (
    generate_docx_from_json,
    generate_pdf_from_json,
    format_resume_as_text,
    set_cell_shading,
    add_horizontal_line,
)

# Salary and domain filtering
from modules.analysis.match_analysis import (
    filter_jobs_by_domains,
    filter_jobs_by_salary,
    extract_salary_from_text,
    extract_salary_from_text_regex,
    calculate_salary_band,
)

# Database queries
from database.queries import (
    get_all_job_seekers,
    get_all_job_seekers_formatted,
    get_all_jobs_for_matching,
    get_all_jobs_for_matching_tuples,
    get_jobs_for_interview,
    get_job_seeker_profile,
    get_job_seeker_profile_tuple,
    save_job_seeker_info,
    save_head_hunter_job,
    init_database,
    init_head_hunter_database,
    get_job_seeker_search_fields,
)

# Also import from api_clients for backward compatibility
from modules.utils.api_clients import (
    IndeedScraperAPI,
    APIMEmbeddingGenerator,
    AzureOpenAITextGenerator,
    get_token_tracker,
    get_embedding_generator,
    get_text_generator,
    get_job_scraper,
)

# Import Indeed API from services (new location)
from services.indeed_api import (
    IndeedJobScraper,
    get_indeed_job_scraper,
)

from modules.utils.helpers import api_call_with_retry


# ============================================================================
# CACHED RESOURCE FUNCTIONS (re-exported for backward compatibility)
# ============================================================================

def get_sentence_transformer_model():
    """Get cached SentenceTransformer model - only loads once."""
    return _get_sentence_transformer_model_cached(Config.MODEL_NAME)


def get_pinecone_client():
    """Get cached Pinecone client - only initializes once."""
    return _get_pinecone_client_cached(Config.PINECONE_API_KEY)


# ============================================================================
# JOBSEEKER BACKEND - MAIN ORCHESTRATOR
# ============================================================================

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
        import openai
        import json
        
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

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        try:
            return json.loads(response.choices[0].message.content)
        except Exception:
            return {}


# ============================================================================
# JOB MATCHER BACKEND (alternate implementation for JSearch API)
# ============================================================================

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
            API_KEY = "your_jsearch_api_key_here"
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


# ============================================================================
# MATCH STATISTICS UI
# ============================================================================

def show_match_statistics():
    """Show match statistics in Streamlit UI."""
    st.header("📊 Match Statistics")

    jobs = get_all_jobs_for_matching_tuples()
    seekers = get_all_job_seekers_formatted()

    if not jobs or not seekers:
        st.info("No statistics data available")
        return

    # Industry distribution
    st.subheader("🏭 Industry Distribution")
    industry_counts = {}
    for job in jobs:
        industry = job[6] if job[6] else "Not Specified"
        industry_counts[industry] = industry_counts.get(industry, 0) + 1

    for industry, count in industry_counts.items():
        percentage = (count / len(jobs)) * 100
        st.write(f"• **{industry}:** {count} Positions ({percentage:.1f}%)")

    # Experience Level Distribution
    st.subheader("🎯 Experience Level Distribution")
    experience_counts = {}
    for job in jobs:
        experience = job[11] if len(job) > 11 and job[11] else "Not Specified"
        experience_counts[experience] = experience_counts.get(experience, 0) + 1

    for exp, count in experience_counts.items():
        st.write(f"• **{exp}:** {count} Positions")


def show_instructions():
    """Display usage instructions in Streamlit UI."""
    st.header("📖 Instructions")

    st.info("""
    **Recruitment Match Instructions:**

    1. **Select Position**: Choose a position from the positions published by the headhunter module
    2. **Set Conditions**: Adjust the minimum match score and display count
    3. **Start Matching**: The system will automatically analyze the match between all job seekers and the position
    4. **View Results**: View detailed match analysis report
    5. **Take Action**: Contact candidates, schedule interviews

    **Matching Algorithm Based on:**
    • Skill Match (Hard Skills)
    • Experience Fit (Work Experience Years)
    • Industry Relevance (Industry Preferences)
    • Location Match (Work Location Preferences)
    • Comprehensive Assessment Analysis

    **Data Sources:**
    • Position Information: Positions published by Head Hunter module
    • Job Seeker Information: Information filled in Job Seeker page
    """)


# ============================================================================
# EXPORTS (for backward compatibility)
# ============================================================================

__all__ = [
    # Main backends
    'JobSeekerBackend',
    'JobMatcherBackend',
    # Resume parser
    'ResumeParser',
    'GPT4JobRoleDetector',
    # Job matcher
    'JobMatcher',
    # LinkedIn searcher
    'LinkedInJobSearcher',
    'get_linkedin_job_searcher',
    # Indeed scraper (both old and new names)
    'IndeedScraperAPI',
    'IndeedJobScraper',
    'get_indeed_job_scraper',
    # Rate limiting
    'TokenUsageTracker',
    'RateLimiter',
    # Match functions
    'analyze_match_simple',
    'calculate_match_scores',
    'calculate_job_match_score',
    # Filter functions
    'filter_jobs_by_domains',
    'filter_jobs_by_salary',
    'calculate_salary_band',
    # Salary extraction
    'extract_salary_from_text',
    'extract_salary_from_text_regex',
    # Profile extraction
    'extract_relevant_resume_sections',
    'extract_structured_profile',
    'generate_tailored_resume',
    # Resume formatters
    'generate_docx_from_json',
    'generate_pdf_from_json',
    'format_resume_as_text',
    'set_cell_shading',
    'add_horizontal_line',
    # Interview functions
    'initialize_interview_session',
    'generate_interview_question',
    'evaluate_answer',
    'generate_final_summary',
    'ai_interview_page',
    # Database functions
    'get_all_job_seekers',
    'get_all_job_seekers_formatted',
    'get_all_jobs_for_matching',
    'get_all_jobs_for_matching_tuples',
    'get_jobs_for_interview',
    'get_job_seeker_profile',
    'get_job_seeker_profile_tuple',
    'save_job_seeker_info',
    'save_head_hunter_job',
    'init_database',
    'init_head_hunter_database',
    # API helpers
    'api_call_with_retry',
    'get_token_tracker',
    'get_embedding_generator',
    'get_text_generator',
    'get_job_scraper',
    # UI functions
    'show_match_statistics',
    'show_instructions',
    # Cache functions
    'get_sentence_transformer_model',
    'get_pinecone_client',
]
