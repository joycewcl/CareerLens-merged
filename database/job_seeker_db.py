# database/job_seeker_db.py
"""
Job Seeker Database operations.

This module handles all database operations for job_seeker.db:
- Storing job seeker profiles
- Retrieving profiles by job_seeker_id
- Profile management

Data Flow Step 1:
    Resume Upload → GPT-4 Parsing → Auto-fill Form → Save to job_seeker.db
"""

import sqlite3
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Dict, List
from datetime import datetime
import uuid


# Database path constant
DB_PATH_JOB_SEEKER = "database/job_seeker.db"


class JobSeekerDB:
    """Job seeker database operations for job_seeker.db."""
    
    def __init__(self, db_path: str = ""):
        self.db_path = Path(db_path or DB_PATH_JOB_SEEKER)
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
        """Initialize database schema."""
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS job_seekers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_seeker_id TEXT UNIQUE NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    education_level TEXT,
                    major TEXT,
                    graduation_status TEXT,
                    university_background TEXT,
                    languages TEXT,
                    certificates TEXT,
                    hard_skills TEXT,
                    soft_skills TEXT,
                    work_experience TEXT,
                    project_experience TEXT,
                    detailed_experience TEXT,
                    location_preference TEXT,
                    industry_preference TEXT,
                    salary_expectation TEXT,
                    benefits_expectation TEXT,
                    primary_role TEXT,
                    simple_search_terms TEXT
                )
            """)
            # Add indexes
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_job_seeker_id 
                ON job_seekers(job_seeker_id)
            """)
            
            # Migration: Ensure detailed_experience column exists (for existing databases)
            try:
                conn.execute("ALTER TABLE job_seekers ADD COLUMN detailed_experience TEXT")
            except Exception:
                # Column likely already exists or table was just created with it
                pass
    
    @staticmethod
    def generate_job_seeker_id() -> str:
        """Generate unique job seeker ID (e.g., JS_ABC12345)."""
        return f"JS_{uuid.uuid4().hex[:8].upper()}"
    
    def save_profile(self, profile: Dict) -> str:
        """Save job seeker profile to database.
        
        Args:
            profile: Dictionary containing profile fields
            
        Returns:
            The job_seeker_id of the saved profile
        """
        job_seeker_id = profile.get('job_seeker_id') or self.generate_job_seeker_id()
        timestamp = profile.get('timestamp') or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO job_seekers (
                    job_seeker_id, timestamp, education_level, major, graduation_status,
                    university_background, languages, certificates, hard_skills, soft_skills,
                    work_experience, project_experience, detailed_experience, location_preference, industry_preference,
                    salary_expectation, benefits_expectation, primary_role, simple_search_terms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job_seeker_id,
                timestamp,
                profile.get('education_level', ''),
                profile.get('major', ''),
                profile.get('graduation_status', ''),
                profile.get('university_background', ''),
                profile.get('languages', ''),
                profile.get('certificates', ''),
                profile.get('hard_skills', ''),
                profile.get('soft_skills', ''),
                profile.get('work_experience', ''),
                profile.get('project_experience', ''),
                profile.get('detailed_experience', ''),
                profile.get('location_preference', ''),
                profile.get('industry_preference', ''),
                profile.get('salary_expectation', ''),
                profile.get('benefits_expectation', ''),
                profile.get('primary_role', ''),
                profile.get('simple_search_terms', '')
            ))
            return job_seeker_id
    
    def get_profile(self, job_seeker_id: str) -> Optional[Dict]:
        """Get job seeker profile by ID.
        
        Args:
            job_seeker_id: The job seeker's unique ID
            
        Returns:
            Profile dictionary or None if not found
        """
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM job_seekers WHERE job_seeker_id = ?
            """, (job_seeker_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_latest_job_seeker_id(self) -> Optional[str]:
        """Get the latest job_seeker_id."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT job_seeker_id FROM job_seekers ORDER BY id DESC LIMIT 1"
            )
            row = cursor.fetchone()
            return row['job_seeker_id'] if row else None
    
    def get_latest_profile(self) -> Optional[Dict]:
        """Get complete data of the latest job seeker."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM job_seekers ORDER BY id DESC LIMIT 1"
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_all_profiles(self) -> List[Dict]:
        """Get all job seeker profiles."""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM job_seekers")
            return [dict(row) for row in cursor.fetchall()]
    
    def get_search_fields(self, job_seeker_id: str) -> Optional[Dict]:
        """Get job seeker search fields by ID.
        
        Args:
            job_seeker_id: The job seeker's unique ID
            
        Returns:
            Dictionary with search-relevant fields or None
        """
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    education_level, major, graduation_status, university_background,
                    languages, certificates, hard_skills, soft_skills,
                    work_experience, project_experience, detailed_experience, location_preference,
                    industry_preference, salary_expectation, benefits_expectation,
                    primary_role, simple_search_terms
                FROM job_seekers
                WHERE job_seeker_id = ?
            """, (job_seeker_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return {
                "education_level": row['education_level'] or "",
                "major": row['major'] or "",
                "graduation_status": row['graduation_status'] or "",
                "university_background": row['university_background'] or "",
                "languages": row['languages'] or "",
                "certificates": row['certificates'] or "",
                "hard_skills": row['hard_skills'] or "",
                "soft_skills": row['soft_skills'] or "",
                "work_experience": row['work_experience'] or "",
                "project_experience": row['project_experience'] or "",
                "detailed_experience": row['detailed_experience'] or "",
                "location_preference": row['location_preference'] or "",
                "industry_preference": row['industry_preference'] or "",
                "salary_expectation": row['salary_expectation'] or "",
                "benefits_expectation": row['benefits_expectation'] or "",
                "primary_role": row['primary_role'] or "",
                "simple_search_terms": row['simple_search_terms'] or "",
            }
    
    def update_profile(self, job_seeker_id: str, updates: Dict) -> bool:
        """Update an existing job seeker profile.
        
        Args:
            job_seeker_id: The job seeker's unique ID
            updates: Dictionary of fields to update
            
        Returns:
            True if updated, False if not found
        """
        if not updates:
            return False
        
        # Build dynamic UPDATE statement
        set_clauses = []
        values = []
        for key, value in updates.items():
            if key not in ('id', 'job_seeker_id'):  # Don't allow updating these
                set_clauses.append(f"{key} = ?")
                values.append(value)
        
        if not set_clauses:
            return False
        
        values.append(job_seeker_id)
        
        with self.get_connection() as conn:
            cursor = conn.execute(f"""
                UPDATE job_seekers 
                SET {', '.join(set_clauses)}
                WHERE job_seeker_id = ?
            """, values)
            return cursor.rowcount > 0
    
    def delete_profile(self, job_seeker_id: str) -> bool:
        """Delete a job seeker profile.
        
        Args:
            job_seeker_id: The job seeker's unique ID
            
        Returns:
            True if deleted, False if not found
        """
        with self.get_connection() as conn:
            cursor = conn.execute("""
                DELETE FROM job_seekers WHERE job_seeker_id = ?
            """, (job_seeker_id,))
            return cursor.rowcount > 0
