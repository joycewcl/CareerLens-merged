"""Analysis module for job match analysis"""
from .match_analysis import (
    calculate_salary_band,
    filter_jobs_by_domains,
    filter_jobs_by_salary
)

__all__ = [
    'calculate_salary_band',
    'filter_jobs_by_domains',
    'filter_jobs_by_salary'
]
