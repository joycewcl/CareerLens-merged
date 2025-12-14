"""
Database query functions.
Consolidates all DB access from backend.py
"""
from typing import List, Dict, Optional
from .models import JobSeekerDB, HeadhunterDB

# Initialize singletons
_job_seeker_db = None
_headhunter_db = None


def get_job_seeker_db() -> JobSeekerDB:
    """Get job seeker database instance (singleton)."""
    global _job_seeker_db
    if _job_seeker_db is None:
        _job_seeker_db = JobSeekerDB()
    return _job_seeker_db


def get_headhunter_db() -> HeadhunterDB:
    """Get headhunter database instance (singleton)."""
    global _headhunter_db
    if _headhunter_db is None:
        _headhunter_db = HeadhunterDB()
    return _headhunter_db


# Query functions that were in backend.py
def get_all_job_seekers() -> List[Dict]:
    """Get all job seekers."""
    return get_job_seeker_db().get_all_profiles()


def get_job_seeker_profile(job_seeker_id: str) -> Optional[Dict]:
    """Get specific job seeker profile."""
    return get_job_seeker_db().get_profile(job_seeker_id)


def get_all_jobs_for_matching() -> List[Dict]:
    """Get all jobs for matching."""
    return get_headhunter_db().get_all_jobs()


def save_job_seeker_info(profile: Dict) -> str:
    """Save job seeker information."""
    return get_job_seeker_db().save_profile(profile)


def save_head_hunter_job(job: Dict) -> int:
    """Save headhunter job posting."""
    return get_headhunter_db().save_job(job)
