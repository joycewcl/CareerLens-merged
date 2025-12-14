"""
Database package for CareerLens.
"""
from database.models import (
    DatabaseConnection,
    JobSeekerDB,
    HeadhunterDB,
    DB_PATH_JOB_SEEKER,
    DB_PATH_HEAD_HUNTER,
)
from database.queries import (
    get_job_seeker_db,
    get_headhunter_db,
    get_all_job_seekers,
    get_job_seeker_profile,
    get_all_jobs_for_matching,
    save_job_seeker_info,
    save_head_hunter_job,
)

__all__ = [
    'DatabaseConnection',
    'JobSeekerDB',
    'HeadhunterDB',
    'DB_PATH_JOB_SEEKER',
    'DB_PATH_HEAD_HUNTER',
    'get_job_seeker_db',
    'get_headhunter_db',
    'get_all_job_seekers',
    'get_job_seeker_profile',
    'get_all_jobs_for_matching',
    'save_job_seeker_info',
    'save_head_hunter_job',
]
