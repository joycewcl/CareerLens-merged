# core/job_seeker_flow.py
"""
Job Seeker Data Flow Orchestration Module.

This module provides centralized functions for the 3-step job seeker data flow:

Step 1: Resume Upload & Profile Creation
    Job Seeker uploads resume
         ↓
    core/resume_parser.py (GPT-4 parsing)
         ↓
    Auto-fill profile form
         ↓
    Save to job_seeker.db
         ↓
    Generate job_seeker_id (e.g., JS_ABC12345)

Step 2: Job Matching & API Storage
    Job Seeker sets search filters/preferences
         ↓
    services/indeed_api.py → Fetch jobs from API
         ↓
    core/semantic_search.py → Generate embeddings
         ↓
    Calculate cosine similarity with job_seeker profile
         ↓
    Rank jobs by match score
         ↓
    Store matched jobs in job_post_API.db
       (All matched jobs are saved)

Step 3: AI Resume & Interview Use Cases
    Tailored Resume Generation:
       ↓ Retrieve from job_post_API.db
       ↓ WHERE job_seeker_id = current_user
       ↓ Get specific job details
       ↓ Merge with job_seeker.db profile
       ↓ Generate optimized resume

    AI Mock Interview:
       ↓ Retrieve job from job_post_API.db
       ↓ Get required_skills, job_description
       ↓ Retrieve candidate profile from job_seeker.db
       ↓ Generate relevant interview questions
"""

from typing import Dict, List, Optional, Tuple
import uuid
from datetime import datetime


# Match score threshold (percentage) - jobs below this won't be saved
MATCH_SCORE_THRESHOLD = 60.0


# =============================================================================
# STEP 1: Resume Upload & Profile Creation
# =============================================================================

def process_resume_and_create_profile(
    resume_file,
    filename: str,
    config=None
) -> Tuple[Optional[str], Optional[Dict], Optional[Dict]]:
    """
    Step 1: Process resume upload and create job seeker profile.
    
    Flow:
        1. Extract text from resume file
        2. Parse with GPT-4 to extract structured data
        3. Generate job_seeker_id
        4. Return data for auto-filling profile form
    
    Args:
        resume_file: File-like object containing resume
        filename: Original filename with extension
        config: Optional config object
        
    Returns:
        Tuple of (job_seeker_id, resume_data, ai_analysis)
        - job_seeker_id: Generated ID like JS_ABC12345
        - resume_data: Extracted raw resume data
        - ai_analysis: AI analysis results for auto-filling form
    """
    from core.resume_parser import ResumeParser, GPT4JobRoleDetector
    from database.models import JobSeekerDB
    
    parser = ResumeParser()
    
    try:
        # Extract text from resume
        resume_data = parser.parse_resume(resume_file, filename)
        
        if not resume_data or not resume_data.get('raw_text'):
            return None, None, None
        
        # Analyze with GPT-4
        detector = GPT4JobRoleDetector(config)
        ai_analysis = detector.analyze_resume_for_job_roles(resume_data)
        
        # Generate job_seeker_id
        job_seeker_id = JobSeekerDB.generate_job_seeker_id()
        
        return job_seeker_id, resume_data, ai_analysis
        
    except Exception as e:
        print(f"Error processing resume: {e}")
        return None, None, None


def save_job_seeker_profile(
    job_seeker_id: str,
    profile_data: Dict
) -> bool:
    """
    Save job seeker profile to job_seeker.db.
    
    Args:
        job_seeker_id: The job seeker's unique ID
        profile_data: Dictionary with profile fields
        
    Returns:
        True if saved successfully, False otherwise
    """
    from database import save_job_seeker_info
    
    # Ensure job_seeker_id is in the profile data
    profile_data['job_seeker_id'] = job_seeker_id
    
    try:
        saved_id = save_job_seeker_info(profile_data)
        return saved_id is not None
    except Exception as e:
        print(f"Error saving profile: {e}")
        return False


def get_job_seeker_profile(job_seeker_id: str) -> Optional[Dict]:
    """
    Get job seeker profile from job_seeker.db.
    
    Args:
        job_seeker_id: The job seeker's unique ID
        
    Returns:
        Profile dictionary or None if not found
    """
    from database import get_job_seeker_profile
    return get_job_seeker_profile(job_seeker_id)


# =============================================================================
# STEP 2: Job Matching & API Storage
# =============================================================================

def search_and_match_jobs(
    job_seeker_id: str,
    search_query: str,
    location: str = "Hong Kong",
    country: str = "hk",
    max_jobs: int = 25,
    job_type: str = "fulltime",
    match_threshold: float = MATCH_SCORE_THRESHOLD,
    force_refresh: bool = False
) -> List[Dict]:
    """
    Step 2: Search jobs from API, match with profile, and store in job_post_API.db.
    
    Flow:
        1. Fetch jobs from Indeed API
        2. Generate embeddings for jobs
        3. Calculate cosine similarity with job seeker profile
        4. Rank jobs by match score
        5. Store jobs above threshold in job_post_API.db
    
    Args:
        job_seeker_id: The job seeker's unique ID
        search_query: Job search keywords
        location: Location to search in
        country: Country code (hk, us, gb, etc.)
        max_jobs: Maximum number of jobs to fetch
        job_type: Employment type (fulltime, parttime, contract)
        match_threshold: Minimum match score to save (0-100)
        force_refresh: Whether to bypass cache
        
    Returns:
        List of matched job dictionaries with scores
    """
    from database import get_job_seeker_profile
    from database.queries import get_matched_jobs_db
    
    # Get job seeker profile
    profile = get_job_seeker_profile(job_seeker_id)
    if not profile:
        print(f"No profile found for job seeker: {job_seeker_id}")
        return []
    
    # Fetch and match jobs
    matched_jobs = _fetch_and_match_jobs(
        profile=profile,
        search_query=search_query,
        location=location,
        country=country,
        max_jobs=max_jobs,
        job_type=job_type,
        force_refresh=force_refresh
    )
    
    if not matched_jobs:
        return []
    
    # Filter by threshold and prepare for storage
    jobs_to_store = []
    for job in matched_jobs:
        combined_score = job.get('combined_score', job.get('combined_match_score', 0))
        
        if combined_score >= match_threshold:
            # Prepare job data for storage
            job_record = _prepare_job_for_storage(job_seeker_id, job)
            jobs_to_store.append(job_record)
    
    # Store matched jobs in job_post_API.db
    if jobs_to_store:
        _store_matched_jobs_batch(jobs_to_store)
    
    return matched_jobs


def _fetch_and_match_jobs(
    profile: Dict,
    search_query: str,
    location: str,
    country: str,
    max_jobs: int,
    job_type: str,
    force_refresh: bool
) -> List[Dict]:
    """
    Internal function to fetch jobs from API and calculate match scores.
    
    Args:
        profile: Job seeker profile dictionary
        search_query: Search keywords
        location: Location to search
        country: Country code
        max_jobs: Maximum jobs to fetch
        job_type: Employment type
        force_refresh: Whether to bypass cache
        
    Returns:
        List of jobs with match scores
    """
    try:
        from core.semantic_search import SemanticJobSearch, fetch_jobs_with_cache
        from utils import get_embedding_generator, get_job_scraper
        from utils.config import _determine_index_limit
        from ui.components.dashboard import calculate_match_scores
    except ImportError as e:
        print(f"Import error in _fetch_and_match_jobs: {e}")
        return []
    
    # Get job scraper
    scraper = get_job_scraper()
    if scraper is None:
        print("Job scraper not configured")
        return []
    
    # Fetch jobs
    jobs = fetch_jobs_with_cache(
        scraper,
        search_query,
        location=location,
        max_rows=max_jobs,
        job_type=job_type,
        country=country,
        force_refresh=force_refresh
    )
    
    if not jobs:
        return []
    
    # Get embedding generator
    embedding_gen = get_embedding_generator()
    if embedding_gen is None:
        print("Embedding generator not configured")
        return []
    
    # Initialize semantic search and index jobs
    search_engine = SemanticJobSearch(embedding_gen)
    jobs_to_index_limit = _determine_index_limit(len(jobs), min(10, len(jobs)))
    search_engine.index_jobs(jobs, max_jobs_to_index=jobs_to_index_limit)
    
    # Build resume query text from profile
    hard_skills = profile.get('hard_skills', '')
    resume_query = f"""
    {profile.get('primary_role', '')}
    {profile.get('simple_search_terms', '')}
    {hard_skills}
    {profile.get('soft_skills', '')}
    {profile.get('work_experience', '')}
    {profile.get('project_experience', '')}
    """
    
    # Search and rank jobs
    results = search_engine.search(query=resume_query, top_k=len(jobs))
    
    if not results:
        return []
    
    # Calculate match scores
    matched_jobs = calculate_match_scores(results, hard_skills)
    
    # Sort by combined score
    matched_jobs.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
    
    return matched_jobs


def _prepare_job_for_storage(job_seeker_id: str, job: Dict) -> Dict:
    """
    Prepare a matched job record for storage in job_post_API.db.
    
    Args:
        job_seeker_id: The job seeker's ID
        job: Matched job dictionary with scores
        
    Returns:
        Dictionary ready for database storage
    """
    # Handle both nested 'job' structure and flat structure
    job_data = job.get('job', job)
    
    # Generate unique job_id
    job_url = job_data.get('url', '')
    job_title = job_data.get('title', '')
    job_company = job_data.get('company', '')
    job_id = f"JOB_{uuid.uuid4().hex[:12].upper()}"
    
    # Extract scores from result
    cosine_score = job.get('semantic_score', job.get('similarity_score', 0)) / 100.0
    match_percentage = int(job.get('combined_score', job.get('combined_match_score', 0)))
    skill_match_score = job.get('skill_match_percentage', job.get('skill_match_score', 0)) / 100.0
    
    # Get matched and missing skills
    matched_skills = job.get('matched_skills', [])
    missing_skills = job.get('missing_skills', job_data.get('skills', []))
    
    # Handle skills lists
    required_skills = job_data.get('skills', [])
    if isinstance(required_skills, list):
        required_skills_str = ', '.join(required_skills[:20])
    else:
        required_skills_str = str(required_skills)
    
    if isinstance(matched_skills, list):
        matched_skills_str = ', '.join(matched_skills[:20])
    else:
        matched_skills_str = str(matched_skills)
    
    if isinstance(missing_skills, list):
        missing_skills_str = ', '.join(missing_skills[:10])
    else:
        missing_skills_str = str(missing_skills)
    
    return {
        'job_seeker_id': job_seeker_id,
        'job_id': job_id,
        'job_title': job_title,
        'company_name': job_company,
        'location': job_data.get('location', ''),
        'job_description': job_data.get('description', '')[:5000],  # Truncate for storage
        'required_skills': required_skills_str,
        'preferred_skills': '',
        'experience_required': job_data.get('experience_level', ''),
        'salary_min': None,
        'salary_max': None,
        'employment_type': job_data.get('job_type', 'Full-time'),
        'industry': job_data.get('industry', ''),
        'posted_date': job_data.get('posted_date', ''),
        'application_url': job_url,
        'cosine_similarity_score': cosine_score,
        'match_percentage': match_percentage,
        'skill_match_score': skill_match_score,
        'experience_match_score': None,
        'matched_skills': matched_skills_str,
        'missing_skills': missing_skills_str
    }


def _store_matched_jobs_batch(jobs: List[Dict]) -> int:
    """
    Store matched jobs in job_post_API.db.
    
    Args:
        jobs: List of job dictionaries prepared for storage
        
    Returns:
        Number of jobs saved
    """
    from database import save_matched_jobs_batch
    return save_matched_jobs_batch(jobs)


def get_matched_jobs_for_seeker(
    job_seeker_id: str,
    min_score: float = MATCH_SCORE_THRESHOLD / 100.0,
    limit: int = 50
) -> List[Dict]:
    """
    Get all matched jobs for a job seeker from job_post_API.db.
    
    Args:
        job_seeker_id: The job seeker's ID
        min_score: Minimum cosine similarity score (0-1)
        limit: Maximum number of results
        
    Returns:
        List of matched job dictionaries
    """
    from database import get_matched_jobs_for_seeker
    return get_matched_jobs_for_seeker(job_seeker_id, min_score, limit)


def get_top_matched_jobs(job_seeker_id: str, limit: int = 10) -> List[Dict]:
    """
    Get top matched jobs for a job seeker by score.
    
    Args:
        job_seeker_id: The job seeker's ID
        limit: Maximum number of results
        
    Returns:
        List of top matched job dictionaries
    """
    from database import get_top_job_matches
    return get_top_job_matches(job_seeker_id, limit)


# =============================================================================
# STEP 3: AI Resume & Interview Use Cases
# =============================================================================

def get_job_for_resume_tailoring(
    job_seeker_id: str,
    job_id: str
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Step 3a: Retrieve job and profile for tailored resume generation.
    
    Flow:
        1. Retrieve job from job_post_API.db WHERE job_seeker_id = current_user
        2. Retrieve candidate profile from job_seeker.db
        3. Return both for resume generation
    
    Args:
        job_seeker_id: The job seeker's ID
        job_id: The job's ID in job_post_API.db
        
    Returns:
        Tuple of (job_data, profile_data)
    """
    from database import get_matched_job, get_job_seeker_profile
    
    # Get job from job_post_API.db
    job_data = get_matched_job(job_id)
    
    if not job_data:
        return None, None
    
    # Verify job belongs to this job seeker
    if job_data.get('job_seeker_id') != job_seeker_id:
        print(f"Job {job_id} does not belong to job seeker {job_seeker_id}")
        return None, None
    
    # Get profile from job_seeker.db
    profile_data = get_job_seeker_profile(job_seeker_id)
    
    return job_data, profile_data


def generate_tailored_resume_for_job(
    job_seeker_id: str,
    job_id: str,
    raw_resume_text: str = None,
    config=None
) -> Optional[Dict]:
    """
    Generate a tailored resume for a specific job.
    
    Flow:
        1. Retrieve job and profile data
        2. Generate optimized resume using AI
        3. Return structured resume data
    
    Args:
        job_seeker_id: The job seeker's ID
        job_id: The job's ID in job_post_API.db
        raw_resume_text: Optional raw resume text for reference
        config: Optional config object
        
    Returns:
        Generated resume data dictionary or None
    """
    from core.resume_parser import generate_tailored_resume
    
    # Get job and profile
    job_data, profile_data = get_job_for_resume_tailoring(job_seeker_id, job_id)
    
    if not job_data or not profile_data:
        return None
    
    # Transform job_data to format expected by generate_tailored_resume
    job_posting = {
        'title': job_data.get('job_title', ''),
        'company': job_data.get('company_name', ''),
        'description': job_data.get('job_description', ''),
        'skills': [s.strip() for s in job_data.get('required_skills', '').split(',') if s.strip()],
        'location': job_data.get('location', ''),
        'employment_type': job_data.get('employment_type', '')
    }
    
    # Transform profile_data to format expected by generate_tailored_resume
    user_profile = {
        'name': profile_data.get('name', f"Job Seeker {job_seeker_id}"),
        'email': profile_data.get('email', ''),
        'phone': profile_data.get('phone', ''),
        'location': profile_data.get('location_preference', ''),
        'linkedin': profile_data.get('linkedin', ''),
        'summary': profile_data.get('summary', ''),
        'experience': profile_data.get('detailed_experience') or profile_data.get('work_experience', ''),
        'education': profile_data.get('education_level', '') + ' - ' + profile_data.get('major', ''),
        'skills': profile_data.get('hard_skills', ''),
        'certifications': profile_data.get('certificates', '')
    }
    
    return generate_tailored_resume(user_profile, job_posting, raw_resume_text, config)


def get_job_for_interview(
    job_seeker_id: str,
    job_id: str
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Step 3b: Retrieve job and profile for AI mock interview.
    
    Flow:
        1. Retrieve job from job_post_API.db (required_skills, job_description)
        2. Retrieve candidate profile from job_seeker.db
        3. Return both for interview question generation
    
    Args:
        job_seeker_id: The job seeker's ID
        job_id: The job's ID in job_post_API.db
        
    Returns:
        Tuple of (job_data, profile_data)
    """
    # Same as resume tailoring - reuse the function
    return get_job_for_resume_tailoring(job_seeker_id, job_id)


def get_jobs_for_interview_from_matches(job_seeker_id: str) -> List[Dict]:
    """
    Get all matched jobs for interview selection.
    
    This retrieves jobs from job_post_API.db for the job seeker
    to choose from for mock interviews.
    
    Args:
        job_seeker_id: The job seeker's ID
        
    Returns:
        List of job dictionaries available for interviews
    """
    jobs = get_matched_jobs_for_seeker(job_seeker_id, min_score=0.0, limit=50)
    
    # Format for interview page display
    formatted_jobs = []
    for job in jobs:
        formatted_jobs.append({
            'id': job.get('id'),
            'job_id': job.get('job_id'),
            'job_title': job.get('job_title', ''),
            'company_name': job.get('company_name', ''),
            'job_description': job.get('job_description', ''),
            'required_skills': job.get('required_skills', ''),
            'industry': job.get('industry', ''),
            'experience_required': job.get('experience_required', ''),
            'match_percentage': job.get('match_percentage', 0),
            'location': job.get('location', '')
        })
    
    return formatted_jobs


def convert_matched_job_to_interview_tuple(job: Dict) -> tuple:
    """
    Convert a matched job dictionary to the tuple format expected by core/interview.py.
    
    The interview module expects tuples with this structure:
    (id, job_title, job_description, main_responsibilities, required_skills,
     company, industry, experience_level)
    
    Args:
        job: Matched job dictionary from job_post_API.db
        
    Returns:
        Tuple in the format expected by interview functions
    """
    return (
        job.get('id', 0),
        job.get('job_title', ''),
        job.get('job_description', ''),
        '',  # main_responsibilities - not stored in matched jobs
        job.get('required_skills', ''),
        job.get('company_name', ''),
        job.get('industry', ''),
        job.get('experience_required', '')
    )


def convert_profile_to_interview_tuple(profile: Dict) -> tuple:
    """
    Convert a job seeker profile dictionary to the tuple format expected by core/interview.py.
    
    The interview module expects tuples with this structure:
    (education_level, work_experience, hard_skills, soft_skills, project_experience)
    
    Args:
        profile: Job seeker profile dictionary from job_seeker.db
        
    Returns:
        Tuple in the format expected by interview functions
    """
    return (
        profile.get('education_level', ''),
        profile.get('work_experience', ''),
        profile.get('hard_skills', ''),
        profile.get('soft_skills', ''),
        profile.get('project_experience', '')
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_current_job_seeker_id() -> Optional[str]:
    """
    Get the current job seeker ID from session state.
    
    Returns:
        Current job_seeker_id or None
    """
    import streamlit as st
    return st.session_state.get('job_seeker_id')


def set_current_job_seeker_id(job_seeker_id: str):
    """
    Set the current job seeker ID in session state.
    
    Args:
        job_seeker_id: The job seeker's ID
    """
    import streamlit as st
    st.session_state.job_seeker_id = job_seeker_id


def clear_matched_jobs_for_seeker(job_seeker_id: str) -> int:
    """
    Clear all matched jobs for a job seeker from job_post_API.db.
    
    Useful when job seeker wants to refresh their matches.
    
    Args:
        job_seeker_id: The job seeker's ID
        
    Returns:
        Number of records deleted
    """
    from database import delete_matches_for_seeker
    return delete_matches_for_seeker(job_seeker_id)


def get_match_statistics(job_seeker_id: str) -> Dict:
    """
    Get matching statistics for a job seeker.
    
    Args:
        job_seeker_id: The job seeker's ID
        
    Returns:
        Dictionary with statistics (total_matches, avg_similarity, etc.)
    """
    from database import get_match_statistics
    return get_match_statistics(job_seeker_id)
