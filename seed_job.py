
import sqlite3
import os
from datetime import datetime, timedelta

# Path to the database
db_path = "/workspace/database/head_hunter_jobs.db"

# Ensure directory exists
os.makedirs(os.path.dirname(db_path), exist_ok=True)

# Sample job data
sample_job = {
    "job_title": "Senior Software Engineer",
    "job_description": "We are looking for an experienced Software Engineer to join our team. You will be responsible for building high-quality, scalable web applications.",
    "main_responsibilities": "- Design and implement scalable web services\n- Collaborate with frontend engineers\n- Mentor junior developers",
    "required_skills": "Python, Django, React, SQL, AWS",
    "client_company": "TechCorp Inc.",
    "industry": "Technology",
    "work_location": "Remote",
    "work_type": "Remote",
    "company_size": "100-500",
    "employment_type": "Full-time",
    "experience_level": "Senior (5+ years)",
    "visa_support": "No",
    "min_salary": 120000,
    "max_salary": 160000,
    "currency": "USD",
    "benefits": "Health insurance, 401k, Remote work",
    "application_method": "Email",
    "job_valid_until": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
}

def seed_database():
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Create table if not exists (simplified schema from head_hunter_db.py)
    c.execute("""
        CREATE TABLE IF NOT EXISTS head_hunter_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            job_title TEXT,
            job_description TEXT,
            main_responsibilities TEXT,
            required_skills TEXT,
            client_company TEXT,
            industry TEXT,
            work_location TEXT,
            work_type TEXT,
            company_size TEXT,
            employment_type TEXT,
            experience_level TEXT,
            visa_support TEXT,
            min_salary REAL,
            max_salary REAL,
            currency TEXT,
            benefits TEXT,
            application_method TEXT,
            job_valid_until TEXT
        )
    """)
    
    # Check if we already have jobs
    c.execute("SELECT COUNT(*) FROM head_hunter_jobs")
    count = c.fetchone()[0]
    
    if count == 0:
        print("Seeding database with sample job...")
        c.execute("""
            INSERT INTO head_hunter_jobs (
                job_title, job_description, main_responsibilities, required_skills,
                client_company, industry, work_location, work_type, company_size,
                employment_type, experience_level, visa_support,
                min_salary, max_salary, currency, benefits, application_method, job_valid_until
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            sample_job['job_title'],
            sample_job['job_description'],
            sample_job['main_responsibilities'],
            sample_job['required_skills'],
            sample_job['client_company'],
            sample_job['industry'],
            sample_job['work_location'],
            sample_job['work_type'],
            sample_job['company_size'],
            sample_job['employment_type'],
            sample_job['experience_level'],
            sample_job['visa_support'],
            sample_job['min_salary'],
            sample_job['max_salary'],
            sample_job['currency'],
            sample_job['benefits'],
            sample_job['application_method'],
            sample_job['job_valid_until']
        ))
        conn.commit()
        print("Sample job added successfully.")
    else:
        print(f"Database already contains {count} jobs.")
    
    conn.close()

if __name__ == "__main__":
    seed_database()
