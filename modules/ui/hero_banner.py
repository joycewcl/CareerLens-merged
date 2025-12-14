"""Hero banner component"""
import streamlit as st
from .styles import get_logo_html


def render_hero_banner(user_profile, matched_jobs=None):
    """Render the Modern Hero banner with personalized welcome message and Base64 logo watermark"""
    user_name = user_profile.get('name', '') if user_profile else ''
    if not user_name or user_name == 'N/A':
        user_name = 'Professional'
    
    if matched_jobs and len(matched_jobs) > 0:
        avg_score = sum(r.get('combined_score', r.get('combined_match_score', 0)) for r in matched_jobs) / len(matched_jobs)
        # Handle both 0-1 scale (old) and 0-100 scale (new) scores
        display_score = int(avg_score) if avg_score > 1 else int(avg_score * 100)
        subtitle = f"Your AI-powered career analysis is ready. We found {len(matched_jobs)} matching opportunities with {display_score}% average fit."
    elif user_profile and user_profile.get('skills'):
        subtitle = "Your profile is loaded. Unlock AI-powered insights to discover your market positioning and best-fit opportunities."
    else:
        subtitle = "Leverage AI-powered insights to discover your market value, skill gaps, and best-fit opportunities in Hong Kong."
    
    logo_html = get_logo_html()
    st.markdown(f"""
    <div class="hero-container">
        {logo_html}
        <div class="hero-content">
            <div style="color: var(--cyan); font-weight: 600; margin-bottom: 8px; font-size: 12px; letter-spacing: 1.5px;">CAREERLENS AI INTELLIGENCE</div>
            <div class="hero-title">Welcome back, {user_name}.</div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
