"""
Job matching analysis functions.
"""
from typing import List, Dict


def filter_jobs_by_domains(jobs: List[Dict], preferred_domains: List[str]) -> List[Dict]:
    """
    Filter jobs by industry domains.
    This is the ONLY place this function should exist.
    """
    if not preferred_domains:
        return jobs
    
    filtered = []
    for job in jobs:
        job_industry = job.get('industry', '').lower()
        if any(domain.lower() in job_industry for domain in preferred_domains):
            filtered.append(job)
    
    return filtered


def filter_jobs_by_salary(
    jobs: List[Dict],
    expected_salary: float,
    tolerance: float = 0.2
) -> List[Dict]:
    """
    Filter jobs by salary expectations.
    This is the ONLY place this function should exist.
    """
    min_acceptable = expected_salary * (1 - tolerance)
    max_acceptable = expected_salary * (1 + tolerance)
    
    filtered = []
    for job in jobs:
        min_salary = job.get('min_salary', 0)
        max_salary = job.get('max_salary', 0)
        
        if min_salary <= max_acceptable and max_salary >= min_acceptable:
            filtered.append(job)
    
    return filtered


def calculate_salary_band(expected: float) -> Dict[str, float]:
    """Calculate acceptable salary range."""
    return {
        'min': expected * 0.8,
        'max': expected * 1.2,
        'ideal': expected
    }
