"""
Page modules for the CareerLens Streamlit application.

This package contains individual page implementations extracted from streamlit_app.py
for better code organization and maintainability.
"""

from .job_seeker_dashboard import main_analyzer_page
from .job_search_page import job_recommendations_page
from .headhunter_dashboard import (
    enhanced_head_hunter_page,
    publish_new_job,
    view_published_jobs,
    show_job_statistics
)
from .recruitment_match_page import (
    recruitment_match_dashboard,
    recruitment_match_page
)
from .ai_interview_page import (
    ai_interview_dashboard,
    show_interview_guidance,
    show_interview_instructions
)
from .resume_tailor_page import tailored_resume_page
from .market_dashboard_page import market_dashboard_page
from .visualizations import (
    create_enhanced_visualizations,
    create_job_comparison_radar
)

__all__ = [
    # Job Seeker Dashboard
    'main_analyzer_page',
    
    # Job Search Page
    'job_recommendations_page',
    
    # Headhunter Dashboard
    'enhanced_head_hunter_page',
    'publish_new_job',
    'view_published_jobs',
    'show_job_statistics',
    
    # Recruitment Match
    'recruitment_match_dashboard',
    'recruitment_match_page',
    
    # AI Interview
    'ai_interview_dashboard',
    'show_interview_guidance',
    'show_interview_instructions',
    
    # Resume Tailor
    'tailored_resume_page',
    
    # Market Dashboard
    'market_dashboard_page',
    
    # Visualizations
    'create_enhanced_visualizations',
    'create_job_comparison_radar',
]
