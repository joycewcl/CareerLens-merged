# database/job_post_api_db.py
"""
Job Post API Database operations.

This module handles all database operations for job_post_API.db:
- Storing matched jobs from Indeed/LinkedIn APIs
- Retrieving matched jobs by job_seeker_id
- Match statistics and analytics

Data Flow Step 2:
    Job Search → Indeed API → Semantic Search → Cosine Similarity → Store in job_post_API.db
    (All matched jobs are saved)

Data Flow Step 3:
    Tailored Resume: Retrieve from job_post_API.db WHERE job_seeker_id = current_user
    AI Interview: Retrieve job details (required_skills, job_description) for question generation
"""

import sqlite3
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Dict, List
from datetime import datetime


# Database path constant
DB_PATH_JOB_POST_API = "database/job_post_API.db"


class MatchedJobsDB:
    """Matched jobs database for job_post_API.db.
    
    This database stores jobs that have been matched with job seekers,
    including their cosine similarity scores and skill match data.
    """
    
    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path or DB_PATH_JOB_POST_API)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_schema(self):
        """Initialize database schema for matched jobs."""
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS matched_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_seeker_id TEXT NOT NULL,
                    
                    -- Job Details from API
                    job_id TEXT UNIQUE NOT NULL,
                    job_title TEXT NOT NULL,
                    company_name TEXT,
                    location TEXT,
                    job_description TEXT,
                    required_skills TEXT,
                    preferred_skills TEXT,
                    experience_required TEXT,
                    salary_min REAL,
                    salary_max REAL,
                    employment_type TEXT,
                    industry TEXT,
                    posted_date TEXT,
                    application_url TEXT,
                    
                    -- Matching Metadata
                    cosine_similarity_score REAL,
                    match_percentage INTEGER,
                    skill_match_score REAL,
                    experience_match_score REAL,
                    matched_skills TEXT,
                    missing_skills TEXT,
                    
                    -- Timestamps
                    matched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (job_seeker_id) REFERENCES job_seeker(job_seeker_id)
                )
            """)
            # Add indexes for efficient querying
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_job_seeker 
                ON matched_jobs(job_seeker_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_similarity 
                ON matched_jobs(cosine_similarity_score DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_match_date 
                ON matched_jobs(matched_at DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_match_percentage
                ON matched_jobs(match_percentage DESC)
            """)
            # Composite indexes for common query patterns (Step 2 improvement)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_job_seeker_match 
                ON matched_jobs(job_seeker_id, match_percentage DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_job_seeker_date 
                ON matched_jobs(job_seeker_id, matched_at DESC)
            """)
    
    def save_matched_job(self, job_data: Dict) -> int:
        """Save a matched job to the database.
        
        Uses UPSERT (INSERT OR UPDATE) to handle duplicate job_ids.
        
        Args:
            job_data: Dictionary containing job and matching data
            
        Returns:
            The ID of the inserted/updated record
        """
        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO matched_jobs (
                    job_seeker_id, job_id, job_title, company_name, location,
                    job_description, required_skills, preferred_skills, experience_required,
                    salary_min, salary_max, employment_type, industry, posted_date,
                    application_url, cosine_similarity_score, match_percentage,
                    skill_match_score, experience_match_score, matched_skills, missing_skills
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    job_seeker_id = excluded.job_seeker_id,
                    job_title = excluded.job_title,
                    company_name = excluded.company_name,
                    location = excluded.location,
                    job_description = excluded.job_description,
                    required_skills = excluded.required_skills,
                    preferred_skills = excluded.preferred_skills,
                    experience_required = excluded.experience_required,
                    salary_min = excluded.salary_min,
                    salary_max = excluded.salary_max,
                    employment_type = excluded.employment_type,
                    industry = excluded.industry,
                    posted_date = excluded.posted_date,
                    application_url = excluded.application_url,
                    cosine_similarity_score = excluded.cosine_similarity_score,
                    match_percentage = excluded.match_percentage,
                    skill_match_score = excluded.skill_match_score,
                    experience_match_score = excluded.experience_match_score,
                    matched_skills = excluded.matched_skills,
                    missing_skills = excluded.missing_skills,
                    last_updated = CURRENT_TIMESTAMP
            """, (
                job_data.get('job_seeker_id'),
                job_data.get('job_id'),
                job_data.get('job_title'),
                job_data.get('company_name'),
                job_data.get('location'),
                job_data.get('job_description'),
                job_data.get('required_skills'),
                job_data.get('preferred_skills'),
                job_data.get('experience_required'),
                job_data.get('salary_min'),
                job_data.get('salary_max'),
                job_data.get('employment_type'),
                job_data.get('industry'),
                job_data.get('posted_date'),
                job_data.get('application_url'),
                job_data.get('cosine_similarity_score'),
                job_data.get('match_percentage'),
                job_data.get('skill_match_score'),
                job_data.get('experience_match_score'),
                job_data.get('matched_skills'),
                job_data.get('missing_skills'),
            ))
            return cursor.lastrowid
    
    def save_matched_jobs_batch(self, jobs: List[Dict]) -> int:
        """Save multiple matched jobs in a batch.
        
        Args:
            jobs: List of job dictionaries
            
        Returns:
            Number of jobs saved
        """
        count = 0
        for job in jobs:
            try:
                self.save_matched_job(job)
                count += 1
            except Exception as e:
                print(f"Error saving job {job.get('job_id')}: {e}")
        return count
    
    def get_matched_job(self, job_id: str) -> Optional[Dict]:
        """Get a matched job by its external job ID.
        
        Args:
            job_id: External job ID from Indeed/LinkedIn
            
        Returns:
            Job dictionary or None
        """
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM matched_jobs WHERE job_id = ?
            """, (job_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_matched_jobs_by_seeker(
        self, 
        job_seeker_id: str, 
        min_score: float = 0.0,
        limit: int = 100
    ) -> List[Dict]:
        """Get all matched jobs for a job seeker.
        
        This is the main query for Step 3 data flow:
        - Tailored Resume: Get jobs for resume generation
        - AI Interview: Get jobs for interview question generation
        
        Args:
            job_seeker_id: The job seeker's ID
            min_score: Minimum cosine similarity score filter (0-1)
            limit: Maximum number of results
            
        Returns:
            List of matched job dictionaries, ordered by match_percentage
        """
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM matched_jobs 
                WHERE job_seeker_id = ? 
                AND (cosine_similarity_score >= ? OR cosine_similarity_score IS NULL)
                ORDER BY match_percentage DESC, cosine_similarity_score DESC
                LIMIT ?
            """, (job_seeker_id, min_score, limit))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_top_matches(
        self, 
        job_seeker_id: str, 
        limit: int = 10
    ) -> List[Dict]:
        """Get top matched jobs for a job seeker by match percentage.
        
        Args:
            job_seeker_id: The job seeker's ID
            limit: Maximum number of results
            
        Returns:
            List of top matched job dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM matched_jobs 
                WHERE job_seeker_id = ?
                ORDER BY match_percentage DESC, cosine_similarity_score DESC
                LIMIT ?
            """, (job_seeker_id, limit))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_recent_matches(
        self, 
        job_seeker_id: str, 
        limit: int = 20
    ) -> List[Dict]:
        """Get recently matched jobs for a job seeker.
        
        Args:
            job_seeker_id: The job seeker's ID
            limit: Maximum number of results
            
        Returns:
            List of recently matched job dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM matched_jobs 
                WHERE job_seeker_id = ?
                ORDER BY matched_at DESC
                LIMIT ?
            """, (job_seeker_id, limit))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_jobs_for_interview(self, job_seeker_id: str) -> List[Dict]:
        """Get matched jobs formatted for interview module.
        
        Step 3b Data Flow:
        - Retrieve job from job_post_API.db
        - Get required_skills, job_description
        - Used for generating relevant interview questions
        
        Args:
            job_seeker_id: The job seeker's ID
            
        Returns:
            List of job dictionaries with interview-relevant fields
        """
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    id, job_id, job_title, job_description, required_skills,
                    company_name, industry, experience_required, match_percentage,
                    location
                FROM matched_jobs 
                WHERE job_seeker_id = ?
                ORDER BY match_percentage DESC
            """, (job_seeker_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_job_for_resume(self, job_seeker_id: str, job_id: str) -> Optional[Dict]:
        """Get a specific job for resume tailoring.
        
        Step 3a Data Flow:
        - Retrieve from job_post_API.db WHERE job_seeker_id = current_user
        - Get specific job details for resume generation
        
        Args:
            job_seeker_id: The job seeker's ID
            job_id: The job's ID
            
        Returns:
            Job dictionary or None if not found/not owned
        """
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM matched_jobs 
                WHERE job_id = ? AND job_seeker_id = ?
            """, (job_id, job_seeker_id))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def delete_matched_job(self, job_id: str) -> bool:
        """Delete a matched job by its external job ID.
        
        Args:
            job_id: External job ID
            
        Returns:
            True if deleted, False if not found
        """
        with self.get_connection() as conn:
            cursor = conn.execute("""
                DELETE FROM matched_jobs WHERE job_id = ?
            """, (job_id,))
            return cursor.rowcount > 0
    
    def delete_matches_for_seeker(self, job_seeker_id: str) -> int:
        """Delete all matched jobs for a job seeker.
        
        Useful for refreshing matches or cleaning up.
        
        Args:
            job_seeker_id: The job seeker's ID
            
        Returns:
            Number of records deleted
        """
        with self.get_connection() as conn:
            cursor = conn.execute("""
                DELETE FROM matched_jobs WHERE job_seeker_id = ?
            """, (job_seeker_id,))
            return cursor.rowcount
    
    def get_match_statistics(self, job_seeker_id: str) -> Dict:
        """Get matching statistics for a job seeker.
        
        Args:
            job_seeker_id: The job seeker's ID
            
        Returns:
            Dictionary with statistics
        """
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_matches,
                    AVG(cosine_similarity_score) as avg_similarity,
                    MAX(cosine_similarity_score) as max_similarity,
                    MIN(cosine_similarity_score) as min_similarity,
                    AVG(match_percentage) as avg_match_percentage,
                    MAX(match_percentage) as max_match_percentage,
                    AVG(skill_match_score) as avg_skill_match,
                    AVG(experience_match_score) as avg_experience_match
                FROM matched_jobs 
                WHERE job_seeker_id = ?
            """, (job_seeker_id,))
            row = cursor.fetchone()
            return dict(row) if row else {}
    
    def get_all_matched_jobs(self) -> List[Dict]:
        """Get all matched jobs in the database.
        
        Returns:
            List of all matched job dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM matched_jobs 
                ORDER BY matched_at DESC
            """)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_unique_job_seekers(self) -> List[str]:
        """Get list of all unique job_seeker_ids with matches.
        
        Returns:
            List of job_seeker_ids
        """
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT DISTINCT job_seeker_id FROM matched_jobs
            """)
            return [row['job_seeker_id'] for row in cursor.fetchall()]
    
    # =========================================================================
    # STEP 0: Cache Check Functions (Improvement #3)
    # =========================================================================
    
    def has_recent_matches(
        self, 
        job_seeker_id: str, 
        max_age_hours: int = 24
    ) -> bool:
        """Check if job seeker has recent matches (Step 0 optimization).
        
        This avoids unnecessary API calls by checking if we have
        cached matches that are still fresh.
        
        Args:
            job_seeker_id: The job seeker's ID
            max_age_hours: Maximum age of matches in hours (default 24)
            
        Returns:
            True if recent matches exist, False otherwise
        """
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) as count FROM matched_jobs 
                WHERE job_seeker_id = ? 
                AND matched_at >= datetime('now', ?)
            """, (job_seeker_id, f'-{max_age_hours} hours'))
            row = cursor.fetchone()
            return row['count'] > 0 if row else False
    
    def get_recent_match_info(
        self, 
        job_seeker_id: str, 
        max_age_hours: int = 24
    ) -> Optional[Dict]:
        """Get info about recent matches for cache decision.
        
        Step 0: Check if recent matches exist (< 24 hours)
            → If yes, use cached matches from job_post_API.db
            → If no, proceed to Step 2 (fetch fresh from API)
        
        Args:
            job_seeker_id: The job seeker's ID
            max_age_hours: Maximum age of matches in hours
            
        Returns:
            Dict with count and newest_match_time, or None
        """
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as count,
                    MAX(matched_at) as newest_match,
                    MIN(matched_at) as oldest_match
                FROM matched_jobs 
                WHERE job_seeker_id = ? 
                AND matched_at >= datetime('now', ?)
            """, (job_seeker_id, f'-{max_age_hours} hours'))
            row = cursor.fetchone()
            if row and row['count'] > 0:
                return {
                    'count': row['count'],
                    'newest_match': row['newest_match'],
                    'oldest_match': row['oldest_match'],
                    'is_fresh': True
                }
            return None
    
    # =========================================================================
    # UNIFIED FUNCTION (Improvement #4)
    # =========================================================================
    
    def get_matched_jobs(
        self,
        job_seeker_id: str,
        min_match: int = 60,
        purpose: str = 'general',
        limit: int = 50
    ) -> List[Dict]:
        """Unified function to get matched jobs with purpose-specific fields.
        
        This is a cleaner API that serves multiple use cases:
        - purpose='general': All fields for display
        - purpose='resume': Fields needed for resume tailoring
        - purpose='interview': Fields needed for interview questions
        
        Args:
            job_seeker_id: The job seeker's ID
            min_match: Minimum match percentage (default 60)
            purpose: 'general', 'resume', or 'interview'
            limit: Maximum number of results
            
        Returns:
            List of matched job dictionaries
        """
        # Define field selections based on purpose
        if purpose == 'interview':
            fields = """
                id, job_id, job_title, job_description, required_skills,
                company_name, industry, experience_required, match_percentage,
                location, matched_skills, missing_skills
            """
        elif purpose == 'resume':
            fields = """
                id, job_id, job_title, job_description, required_skills,
                preferred_skills, company_name, location, employment_type,
                match_percentage, matched_skills, missing_skills, application_url
            """
        else:  # general
            fields = "*"
        
        with self.get_connection() as conn:
            cursor = conn.execute(f"""
                SELECT {fields} FROM matched_jobs 
                WHERE job_seeker_id = ? 
                AND (match_percentage >= ? OR match_percentage IS NULL)
                ORDER BY match_percentage DESC, cosine_similarity_score DESC
                LIMIT ?
            """, (job_seeker_id, min_match, limit))
            return [dict(row) for row in cursor.fetchall()]
    
    # =========================================================================
    # CLEANUP FUNCTIONS (Improvement #5)
    # =========================================================================
    
    def cleanup_old_matches(
        self, 
        job_seeker_id: str, 
        days: int = 30
    ) -> int:
        """Delete matches older than X days to keep DB lean.
        
        Args:
            job_seeker_id: The job seeker's ID
            days: Delete matches older than this many days (default 30)
            
        Returns:
            Number of records deleted
        """
        with self.get_connection() as conn:
            cursor = conn.execute("""
                DELETE FROM matched_jobs 
                WHERE job_seeker_id = ? 
                AND matched_at < datetime('now', ?)
            """, (job_seeker_id, f'-{days} days'))
            return cursor.rowcount
    
    def cleanup_all_old_matches(self, days: int = 30) -> int:
        """Delete all matches older than X days (global cleanup).
        
        Args:
            days: Delete matches older than this many days (default 30)
            
        Returns:
            Number of records deleted
        """
        with self.get_connection() as conn:
            cursor = conn.execute("""
                DELETE FROM matched_jobs 
                WHERE matched_at < datetime('now', ?)
            """, (f'-{days} days',))
            return cursor.rowcount
    
    def cleanup_low_matches(
        self, 
        job_seeker_id: str, 
        min_match: int = 50,
        keep_count: int = 20
    ) -> int:
        """Delete low-quality matches, keeping only top N.
        
        Useful for keeping the database lean while preserving best matches.
        
        Args:
            job_seeker_id: The job seeker's ID
            min_match: Minimum match percentage to keep
            keep_count: Keep at least this many top matches
            
        Returns:
            Number of records deleted
        """
        with self.get_connection() as conn:
            # First, get IDs of top matches to keep
            cursor = conn.execute("""
                SELECT id FROM matched_jobs 
                WHERE job_seeker_id = ?
                ORDER BY match_percentage DESC
                LIMIT ?
            """, (job_seeker_id, keep_count))
            keep_ids = [row['id'] for row in cursor.fetchall()]
            
            if not keep_ids:
                return 0
            
            # Delete matches that are:
            # 1. Below min_match threshold AND
            # 2. Not in the top N matches
            placeholders = ','.join('?' * len(keep_ids))
            cursor = conn.execute(f"""
                DELETE FROM matched_jobs 
                WHERE job_seeker_id = ? 
                AND match_percentage < ?
                AND id NOT IN ({placeholders})
            """, [job_seeker_id, min_match] + keep_ids)
            return cursor.rowcount
    
    def get_database_stats(self) -> Dict:
        """Get overall database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT job_seeker_id) as unique_seekers,
                    AVG(match_percentage) as avg_match_pct,
                    MIN(matched_at) as oldest_match,
                    MAX(matched_at) as newest_match
                FROM matched_jobs
            """)
            row = cursor.fetchone()
            return dict(row) if row else {}
