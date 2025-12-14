"""UI module for Streamlit components"""
from .sidebar import render_sidebar
from .hero_banner import render_hero_banner
from .job_cards import display_job_card
from .user_profile import display_user_profile
from .dashboard import (
    display_market_positioning_profile,
    display_refine_results_section,
    display_ranked_matches_table,
    display_match_breakdown,
    display_skill_matching_matrix,
    calculate_match_scores
)
from .resume_editor import display_resume_generator, render_structured_resume_editor
from .match_feedback import display_match_score_feedback

# Import page modules
from .pages import (
    main_analyzer_page,
    job_recommendations_page,
    enhanced_head_hunter_page,
    publish_new_job,
    view_published_jobs,
    show_job_statistics,
    recruitment_match_dashboard,
    recruitment_match_page,
    ai_interview_dashboard,
    show_interview_guidance,
    show_interview_instructions,
    tailored_resume_page,
    market_dashboard_page,
    create_enhanced_visualizations,
    create_job_comparison_radar,
)

__all__ = [
    # Core UI components
    'render_sidebar',
    'render_hero_banner',
    'display_job_card',
    'display_user_profile',
    'display_market_positioning_profile',
    'display_refine_results_section',
    'display_ranked_matches_table',
    'display_match_breakdown',
    'display_resume_generator',
    'display_skill_matching_matrix',
    'display_match_score_feedback',
    'render_structured_resume_editor',
    'calculate_match_scores',
    
    # Page modules
    'main_analyzer_page',
    'job_recommendations_page',
    'enhanced_head_hunter_page',
    'publish_new_job',
    'view_published_jobs',
    'show_job_statistics',
    'recruitment_match_dashboard',
    'recruitment_match_page',
    'ai_interview_dashboard',
    'show_interview_guidance',
    'show_interview_instructions',
    'tailored_resume_page',
    'market_dashboard_page',
    'create_enhanced_visualizations',
    'create_job_comparison_radar',
]
