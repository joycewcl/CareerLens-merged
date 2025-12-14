"""
CareerLens - AI Career Intelligence Platform
Combined application with multi-page navigation and modular dashboard

WebSocket Stability Notes:
- This application includes multiple mechanisms to prevent WebSocket disconnections:
  1. Chunked sleep operations to send periodic UI updates
  2. Keepalive pings during long-running operations
  3. Progress tracking with automatic connection maintenance
  4. Optimized server configuration in .streamlit/config.toml
  5. Connection state tracking and recovery mechanisms
  6. Automatic reconnection handling via session state preservation
"""
import warnings
import os
import gc
import sys
import time
warnings.filterwarnings('ignore')

# Streamlit Cloud optimization - set before importing streamlit
os.environ['STREAMLIT_LOG_LEVEL'] = 'error'
os.environ['SQLITE_TMPDIR'] = '/tmp'

# Disable ALL Streamlit telemetry/analytics to prevent tracking script loads
# Note: Browser may still show "Tracking Prevention" console messages - this is
# the browser blocking residual analytics attempts, not an app error
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
os.environ['STREAMLIT_GLOBAL_DEVELOPMENT_MODE'] = 'false'

# WebSocket connection timeout settings (affects Streamlit Cloud behavior)
# Lower values = more frequent keepalive pings but more network overhead
os.environ['STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION'] = 'true'
os.environ['STREAMLIT_SERVER_MAX_MESSAGE_SIZE'] = '200'

# Increase recursion limit for complex operations (prevents stack overflow)
sys.setrecursionlimit(3000)

import streamlit as st
import sqlite3
import json
import base64
from typing import List, Dict
from io import BytesIO

# Lazy imports for heavy visualization libraries - only loaded when needed
_pd = None
_plt = None
_go = None
_np = None
_pinecone = None


def _get_pandas():
    """Lazy load pandas"""
    global _pd
    if _pd is None:
        import pandas as pd
        _pd = pd
    return _pd


def _get_matplotlib():
    """Lazy load matplotlib"""
    global _plt
    if _plt is None:
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt


def _get_plotly():
    """Lazy load plotly"""
    global _go
    if _go is None:
        import plotly.graph_objects as go
        _go = go
    return _go


def _get_numpy():
    """Lazy load numpy"""
    global _np
    if _np is None:
        import numpy as np
        _np = np
    return _np

from backend import JobSeekerBackend
from backend import LinkedInJobSearcher
from backend import get_linkedin_job_searcher
from backend import get_all_jobs_for_matching
from backend import get_all_job_seekers
from backend import analyze_match_simple
from backend import show_match_statistics
from backend import show_instructions

from backend import get_jobs_for_interview
from backend import get_job_seeker_profile
from backend import ai_interview_page

# Import How It Works page
from how_it_works import render_how_it_works_page

# CareerLens imports
from backend import (
    TokenUsageTracker,
    filter_jobs_by_domains,
    filter_jobs_by_salary,
    calculate_salary_band,
    extract_salary_from_text,
    IndeedScraperAPI,
    extract_structured_profile,
    generate_tailored_resume,
    generate_docx_from_json,
    generate_pdf_from_json,
    format_resume_as_text
)

from database import JobSeekerDB
from database import HeadhunterDB

# Cache database instances to avoid recreation on every rerun
@st.cache_resource
def get_job_seeker_db():
    return JobSeekerDB()

@st.cache_resource
def get_headhunter_db():
    return HeadhunterDB()

db = get_job_seeker_db()
db2 = get_headhunter_db()

from database import save_job_seeker_info
from database import save_head_hunter_job
from database import init_database
from database import init_head_hunter_database
from database import get_job_seeker_search_fields
from config import Config

import json
from datetime import datetime

# Initialize token tracker in session state
if 'token_tracker' not in st.session_state:
    st.session_state.token_tracker = TokenUsageTracker()


# ============================================================================
# CAREERLENS DASHBOARD FEATURES
# ============================================================================

def display_skill_matching_matrix(user_profile):
    """Display skill matching calculation matrix (from CareerLens)"""
    st.markdown("---")
    st.markdown("### 📊 How Job Ranking Works")
    
    user_skills = user_profile.get('hard_skills', '') if user_profile else ''
    if not user_skills:
        user_skills = user_profile.get('skills', '') if user_profile else ''
    
    if not user_skills:
        st.info("💡 **Skill-Based Ranking**: Jobs are ranked by how many required skills you match. Upload your CV to see your skills analyzed.")
        return
    
    user_skills_list = [s.strip() for s in str(user_skills).split(',') if s.strip()]
    
    if not user_skills_list:
        st.info("💡 **Skill-Based Ranking**: Jobs are ranked by how many required skills you match.")
        return
    
    st.markdown("#### Your Skills")
    skills_display = ", ".join(user_skills_list[:10])
    if len(user_skills_list) > 10:
        skills_display += f" (+{len(user_skills_list) - 10} more)"
    st.markdown(f"**{len(user_skills_list)} skills identified:** {skills_display}")
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        **Ranking Formula:**
        
        ```
        Skill Match Score = Matched Skills / Required Skills
        ```
        
        **Example:**
        - Job requires: Python, SQL, React, Docker
        - You have: Python, SQL, React
        - **Score: 3/4 = 75%**
        """)
    
    with col2:
        st.markdown("""
        **Ranking Logic:**
        
        1. ✅ Jobs are fetched from job boards
        2. 🔍 Your skills are matched against each job
        3. 📊 Jobs sorted by match score (highest first)
        4. 🎯 Top matches appear at the top
        """)


def display_market_positioning_dashboard(matched_jobs, user_profile):
    """Display Dashboard with 3 key metric cards (from CareerLens)"""
    if not matched_jobs:
        return
    
    # Calculate average match score
    avg_match_score = 0
    for result in matched_jobs:
        if isinstance(result, dict):
            score = result.get('combined_score', result.get('combined_match_score', 0))
            avg_match_score += score
    avg_match_score = avg_match_score / len(matched_jobs) if matched_jobs else 0
    
    # Normalize score if it's already a percentage
    if avg_match_score > 1:
        match_score_pct = int(avg_match_score)
    else:
        match_score_pct = int(avg_match_score * 100)
    
    if match_score_pct >= 80:
        match_delta = "Excellent fit"
    elif match_score_pct >= 60:
        match_delta = "Good fit"
    else:
        match_delta = "Room to improve"
    
    # Calculate salary band
    salary_min, salary_max = calculate_salary_band(matched_jobs)
    avg_salary = (salary_min + salary_max) // 2
    
    # Calculate skill gaps
    user_skills = user_profile.get('hard_skills', '') or user_profile.get('skills', '')
    all_job_skills = []
    for result in matched_jobs:
        job = result.get('job', result)
        job_skills = job.get('skills', []) or job.get('matched_skills', [])
        all_job_skills.extend(job_skills)
    
    user_skills_list = [s.lower().strip() for s in str(user_skills).split(',') if s.strip()]
    skill_gaps = set()
    for job_skill in all_job_skills:
        if isinstance(job_skill, str):
            job_skill_lower = job_skill.lower().strip()
            if job_skill_lower and not any(us in job_skill_lower or job_skill_lower in us for us in user_skills_list):
                skill_gaps.add(job_skill_lower)
    
    num_skill_gaps = min(len(skill_gaps), 20)  # Cap at 20
    
    if num_skill_gaps <= 3:
        gap_delta = "Well positioned"
    elif num_skill_gaps <= 7:
        gap_delta = "Some upskilling needed"
    else:
        gap_delta = "Focus on learning"
    
    st.markdown("### 📊 Your Market Position Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🎯 Match Score", f"{match_score_pct}%", match_delta)
    
    with col2:
        st.metric("💰 Est. Salary", f"HKD {avg_salary // 1000}k", "Market rate")
    
    with col3:
        st.metric("📈 Skill Gaps", f"{num_skill_gaps}", gap_delta)
    
    # Show top skill gaps if any
    if skill_gaps:
        with st.expander("🔧 Top Skills to Develop"):
            gap_list = list(skill_gaps)[:10]
            cols = st.columns(2)
            for i, skill in enumerate(gap_list):
                with cols[i % 2]:
                    st.write(f"• {skill}")


def display_resume_generator_ui(job, user_profile, resume_text=None):
    """Display Resume Tailoring UI (from CareerLens)
    
    Now includes Lazy Pass 2: Profile verification runs here before resume generation,
    ensuring accurate dates, company names, and job titles in the tailored resume.
    """
    from modules.resume_upload import verify_profile_data_pass2
    
    st.subheader("✨ AI Resume Tailoring")
    
    st.info(f"**Tailoring resume for:** {job.get('title', 'Unknown')} at {job.get('company', 'Unknown')}")
    
    if st.button("🚀 Generate Tailored Resume", type="primary", use_container_width=True):
        # Use ProgressTracker to maintain WebSocket connection during AI generation
        with ProgressTracker("Tailoring your resume", total_steps=4) as tracker:
            
            # LAZY PASS 2: Verify profile data before resume generation
            tracker.update(1, "Verifying profile accuracy...")
            _websocket_keepalive("Verifying profile data")
            
            verified_profile = verify_profile_data_pass2(user_profile, resume_text)
            
            # Update session state with verified profile
            if verified_profile and verified_profile != user_profile:
                st.session_state.user_profile = verified_profile
                user_profile = verified_profile
            
            tracker.update(2, "Analyzing job requirements...")
            _websocket_keepalive("Analyzing job")
            
            tracker.update(3, "AI is generating tailored content...")
            _websocket_keepalive("AI generation in progress")
            
            resume_data = generate_tailored_resume(user_profile, job, resume_text)
            
            tracker.update(4, "Finalizing resume...")
            _websocket_keepalive("Complete")
            
            if resume_data:
                st.success("✅ Tailored resume generated!")
                st.session_state.generated_resume = resume_data
                
                # Display preview
                st.markdown("### 📄 Resume Preview")
                
                header = resume_data.get('header', {})
                st.markdown(f"**{header.get('name', 'Your Name')}**")
                st.markdown(f"*{header.get('title', 'Professional')}*")
                
                if resume_data.get('summary'):
                    st.markdown("**Summary:**")
                    st.write(resume_data['summary'])
                
                skills = resume_data.get('skills_highlighted', [])
                if skills:
                    st.markdown("**Key Skills:**")
                    st.write(", ".join(skills[:10]))
                
                # Download buttons
                st.markdown("### 📥 Download Options")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    docx_file = generate_docx_from_json(resume_data)
                    if docx_file:
                        st.download_button(
                            "📄 Download DOCX",
                            docx_file,
                            file_name="tailored_resume.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                
                with col2:
                    pdf_file = generate_pdf_from_json(resume_data)
                    if pdf_file:
                        st.download_button(
                            "📑 Download PDF",
                            pdf_file,
                            file_name="tailored_resume.pdf",
                            mime="application/pdf"
                        )
                
                with col3:
                    text_content = format_resume_as_text(resume_data)
                    st.download_button(
                        "📝 Download TXT",
                        text_content,
                        file_name="tailored_resume.txt",
                        mime="text/plain"
                    )
            else:
                st.error("❌ Failed to generate resume. Please try again.")


def display_domain_filter_ui():
    """Display domain/industry filter UI (from CareerLens)"""
    st.markdown("### 🏭 Filter by Industry")
    
    target_domains = st.multiselect(
        "Target Domains",
        options=["FinTech", "ESG & Sustainability", "Data Analytics", "Digital Transformation", 
                "Investment Banking", "Consulting", "Technology", "Healthcare", "Education",
                "Real Estate", "Retail & E-commerce", "Marketing & Advertising", "Legal", 
                "Human Resources", "Operations"],
        default=st.session_state.get('target_domains', []),
        help="Select industries/domains to search for jobs",
        key="domain_filter"
    )
    st.session_state.target_domains = target_domains
    
    salary_expectation = st.slider(
        "Min. Monthly Salary (HKD)",
        min_value=0,
        max_value=150000,
        value=st.session_state.get('salary_expectation', 0),
        step=5000,
        help="Set to 0 to disable salary filtering",
        key="salary_filter"
    )
    st.session_state.salary_expectation = salary_expectation
    
    return target_domains, salary_expectation


def display_token_usage():
    """Display token usage and cost tracking (from CareerLens)"""
    if 'token_tracker' in st.session_state:
        tracker = st.session_state.token_tracker
        summary = tracker.get_summary()
        
        if summary['total_tokens'] > 0:
            with st.expander("📊 API Usage Stats"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Tokens", f"{summary['total_tokens']:,}")
                with col2:
                    st.metric("Embedding Tokens", f"{summary['embedding_tokens']:,}")
                with col3:
                    st.metric("Est. Cost", f"${summary['estimated_cost_usd']:.4f}")


def create_enhanced_visualizations(matched_jobs: List[Dict], job_seeker_data: Dict = None):
    """Create enhanced visualizations for job matching analysis"""
    if not matched_jobs:
        st.warning("No visualization data available - no jobs matched")
        return

    # Lazy load heavy libraries only when visualizations are created
    np = _get_numpy()
    plt = _get_matplotlib()
    pd = _get_pandas()

    st.markdown("---")
    st.subheader("📊 Advanced Match Analysis")
    
    try:
        # 1. Score comparison chart
        st.markdown("### 🎯 Match Score Breakdown")
        
        # Prepare data for top 5 jobs
        top_jobs = matched_jobs[:5]
        jobs_display = [f"Job {i+1}" for i in range(len(top_jobs))]
        combined_scores = [j.get('combined_score', 0) for j in top_jobs]
        semantic_scores = [j.get('semantic_score', 0) for j in top_jobs]
        skill_scores = [j.get('skill_match_percentage', 0) for j in top_jobs]

        # Create detailed score comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Scores by job - improved styling
        x = np.arange(len(jobs_display))
        width = 0.25

        bars1 = ax1.bar(x - width, combined_scores, width, label='Combined', color='#7e22ce', alpha=0.8)
        bars2 = ax1.bar(x, semantic_scores, width, label='Semantic', color='#3b82f6', alpha=0.8)
        bars3 = ax1.bar(x + width, skill_scores, width, label='Skills', color='#10b981', alpha=0.8)

        ax1.set_xlabel('Jobs', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Match Scores by Job Position', fontsize=14, fontweight='bold', pad=20)
        ax1.set_xticks(x)
        ax1.set_xticklabels(jobs_display, fontsize=10)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)

        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # 2. Enhanced skill distribution
        all_skills = []
        skill_categories = {}
        
        for job in matched_jobs:
            skills = job.get('matched_skills', [])
            all_skills.extend(skills)
            
            # Categorize skills
            for skill in skills:
                skill_lower = skill.lower()
                if any(keyword in skill_lower for keyword in ['python', 'java', 'c++', 'javascript', 'sql', 'r']):
                    category = 'Programming'
                elif any(keyword in skill_lower for keyword in ['machine learning', 'ai', 'deep learning', 'nlp', 'computer vision']):
                    category = 'AI/ML'
                elif any(keyword in skill_lower for keyword in ['tableau', 'power bi', 'excel', 'analysis', 'analytics']):
                    category = 'Analytics'
                elif any(keyword in skill_lower for keyword in ['project', 'management', 'leadership', 'team']):
                    category = 'Management'
                elif any(keyword in skill_lower for keyword in ['communication', 'presentation', 'writing', 'english']):
                    category = 'Communication'
                else:
                    category = 'Other'
                
                skill_categories[category] = skill_categories.get(category, 0) + 1

        # Skill frequency chart
        skill_counts = pd.Series(all_skills).value_counts().head(10)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(skill_counts)))
        bars = ax2.barh(range(len(skill_counts)), skill_counts.values, color=colors, alpha=0.8)
        
        ax2.set_yticks(range(len(skill_counts)))
        ax2.set_yticklabels(skill_counts.index, fontsize=10)
        ax2.set_xlabel('Frequency Across All Matched Jobs', fontsize=12, fontweight='bold')
        ax2.set_title('Most Common Matched Skills', fontsize=14, fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2., 
                    str(int(width)), ha='left', va='center', fontsize=9, fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # 3. Job Match Quality Distribution
        st.markdown("### 📈 Job Match Quality Distribution")
        
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Score distribution histogram
        all_scores = [job.get('combined_score', 0) for job in matched_jobs]
        ax3.hist(all_scores, bins=10, color='#8b5cf6', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Combined Match Score (%)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Number of Jobs', fontsize=12, fontweight='bold')
        ax3.set_title('Distribution of Match Scores', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add statistics
        avg_score = np.mean(all_scores)
        ax3.axvline(avg_score, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_score:.1f}%')
        ax3.legend()

        # Skill match vs semantic match scatter plot
        semantic_scores_all = [job.get('semantic_score', 0) for job in matched_jobs]
        skill_scores_all = [job.get('skill_match_percentage', 0) for job in matched_jobs]
        
        scatter = ax4.scatter(semantic_scores_all, skill_scores_all, 
                             c=all_scores, cmap='viridis', alpha=0.7, s=60)
        ax4.set_xlabel('Semantic Match Score (%)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Skill Match Score (%)', fontsize=12, fontweight='bold')
        ax4.set_title('Semantic vs Skill Match Correlation', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Combined Score (%)', fontsize=10, fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        # 4. Detailed Skill Analysis
        st.markdown("### 🔧 Detailed Skill Analysis")
        
        if job_seeker_data:
            candidate_skills = set()
            hard_skills = job_seeker_data.get('hard_skills', '')
            if hard_skills:
                candidate_skills.update([skill.strip().lower() for skill in hard_skills.split(',')])
            
            # Analyze skill coverage
            total_required_skills = set()
            matched_skills_per_job = []
            
            for job in matched_jobs[:5]:  # Top 5 jobs
                required_skills = set(job.get('matched_skills', []))
                total_required_skills.update(required_skills)
                matched_skills_per_job.append(len(required_skills))
            
            skill_coverage = len(total_required_skills) / len(candidate_skills) * 100 if candidate_skills else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Your Unique Skills", len(candidate_skills))
            with col2:
                st.metric("Skills Required by Top Jobs", len(total_required_skills))
            with col3:
                st.metric("Skill Coverage", f"{skill_coverage:.1f}%")

        # 5. Job Quality Indicators
        st.markdown("### 🎯 Job Quality Indicators")
        
        # Calculate various metrics
        high_match_jobs = len([j for j in matched_jobs if j.get('combined_score', 0) >= 80])
        avg_semantic = np.mean([j.get('semantic_score', 0) for j in matched_jobs])
        avg_skill = np.mean([j.get('skill_match_percentage', 0) for j in matched_jobs])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Matches", len(matched_jobs))
        with col2:
            st.metric("High Quality Matches", high_match_jobs)
        with col3:
            st.metric("Avg Semantic Match", f"{avg_semantic:.1f}%")
        with col4:
            st.metric("Avg Skill Match", f"{avg_skill:.1f}%")

        # 6. Recommendations based on analysis
        st.markdown("### 💡 Personalized Recommendations")
        
        if avg_skill < 50:
            st.warning("**Skill Development Opportunity**: Your skill match is relatively low. Consider:")
            st.write("- Focus on learning the most frequently required skills shown above")
            st.write("- Take online courses for high-demand technologies")
            st.write("- Work on projects that demonstrate these skills")
        
        if avg_semantic > avg_skill:
            st.info("**Strength in Role Fit**: Your experience and background are well-aligned with these roles, even if specific skills need development.")
        
        if high_match_jobs >= 3:
            st.success("**Strong Market Position**: You have multiple high-quality matches! Focus on applying to these top positions.")

    except Exception as e:
        st.error(f"Error creating visualizations: {str(e)}")
        st.info("Please try again with different search parameters")

def create_job_comparison_radar(matched_jobs: List[Dict]):
    """Create radar chart for top 3 job comparisons"""
    if len(matched_jobs) < 2:
        return
    
    # Lazy load plotly only when radar chart is created
    go = _get_plotly()
        
    try:
        st.markdown("### 📊 Job Comparison Radar")
        
        # Define comparison categories
        categories = ['Skill Match', 'Role Relevance', 'Experience Fit', 'Location Match', 'Salary Alignment']
        
        # Calculate scores for each category (simplified for demo)
        job_scores = []
        for job in matched_jobs[:3]:
            scores = [
                job.get('skill_match_percentage', 0),
                job.get('semantic_score', 0),
                min(job.get('semantic_score', 0) * 0.8, 100),  # Simulated experience fit
                75,  # Simulated location match
                70   # Simulated salary alignment
            ]
            job_scores.append(scores)
        
        fig = go.Figure()

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, scores in enumerate(job_scores):
            job_title = matched_jobs[i].get('title', f'Job {i+1}')[:25]
            fig.add_trace(go.Scatterpolar(
                r=scores + [scores[0]],  # Close the radar
                theta=categories + [categories[0]],
                fill='toself',
                name=f"{job_title}",
                line=dict(color=colors[i], width=2),
                opacity=0.7
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(size=10)
                ),
                angularaxis=dict(
                    tickfont=dict(size=11)
                )
            ),
            showlegend=True,
            title=dict(
                text="Multi-dimensional Job Comparison",
                x=0.5,
                font=dict(size=16)
            ),
            height=500,
            margin=dict(l=80, r=80, t=80, b=80)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating radar chart: {str(e)}")


# Page config
st.set_page_config(
    page_title="CareerLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "CareerLens - AI Career Intelligence Platform • Hong Kong"
    }
)

# Import modular UI components from modules/ (for Market Dashboard page)
try:
    from modules.utils import _cleanup_session_state, validate_secrets
    from modules.utils.helpers import (
        _chunked_sleep,
        _websocket_keepalive,
        _ensure_websocket_alive,
        ProgressTracker
    )
    from modules.ui.styles import render_styles
    from modules.ui import (
        render_sidebar as modular_render_sidebar,
        render_hero_banner,
        display_resume_generator as modular_display_resume_generator,
        display_market_positioning_profile,
        display_refine_results_section,
        display_ranked_matches_table,
        display_match_breakdown
    )
    MODULES_AVAILABLE = True
    WEBSOCKET_UTILS_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    WEBSOCKET_UTILS_AVAILABLE = False
    # Modules not available - Market Dashboard page will be disabled
    # Provide fallback implementations for WebSocket utilities
    def _websocket_keepalive(message=None, force=False):
        """Fallback no-op implementation"""
        pass
    
    def _ensure_websocket_alive():
        """Fallback no-op implementation"""
        pass
    
    def _chunked_sleep(delay, message_prefix=""):
        """Fallback implementation using regular sleep"""
        import time
        time.sleep(delay)
    
    class ProgressTracker:
        """Fallback implementation without WebSocket keepalive"""
        def __init__(self, description="Processing", total_steps=100, show_progress=True):
            self.description = description
            self.total_steps = total_steps
            self.show_progress = show_progress
            self.current_step = 0
            self.progress_bar = None
        
        def __enter__(self):
            if self.show_progress:
                self.progress_bar = st.progress(0, text=f"⏳ {self.description}...")
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.progress_bar:
                self.progress_bar.empty()
            return False
        
        def update(self, step=None, message=None):
            if step is not None:
                self.current_step = step
            else:
                self.current_step += 1
            progress = min(self.current_step / self.total_steps, 1.0)
            if self.show_progress and self.progress_bar:
                display_message = message or f"⏳ {self.description}... ({int(progress * 100)}%)"
                self.progress_bar.progress(progress, text=display_message)
        
        def set_message(self, message):
            if self.show_progress and self.progress_bar:
                progress = self.current_step / self.total_steps
                self.progress_bar.progress(progress, text=f"⏳ {message}")

# Initialize backend - cached to prevent re-initialization
@st.cache_resource(show_spinner="Loading CareerLens...")
def load_backend():
    return JobSeekerBackend()

backend = load_backend()


# Initialize database - only once using caching (cleaner than session state)
@st.cache_resource
def initialize_databases():
    init_database()
    init_head_hunter_database()
    return True

_db_initialized = initialize_databases()

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "main"

# Additional session state for modular dashboard
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'jobs_cache' not in st.session_state:
    st.session_state.jobs_cache = {}
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {}
if 'generated_resume' not in st.session_state:
    st.session_state.generated_resume = None
if 'selected_job' not in st.session_state:
    st.session_state.selected_job = None
if 'show_resume_generator' not in st.session_state:
    st.session_state.show_resume_generator = False
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = None
if 'resume_embedding' not in st.session_state:
    st.session_state.resume_embedding = None
if 'matched_jobs' not in st.session_state:
    st.session_state.matched_jobs = []
if 'match_score' not in st.session_state:
    st.session_state.match_score = None
if 'missing_keywords' not in st.session_state:
    st.session_state.missing_keywords = None
if 'show_profile_editor' not in st.session_state:
    st.session_state.show_profile_editor = False
if 'use_auto_match' not in st.session_state:
    st.session_state.use_auto_match = False
if 'expanded_job_index' not in st.session_state:
    st.session_state.expanded_job_index = None
if 'industry_filter' not in st.session_state:
    st.session_state.industry_filter = None
if 'salary_min' not in st.session_state:
    st.session_state.salary_min = None
if 'salary_max' not in st.session_state:
    st.session_state.salary_max = None
if 'selected_job_index' not in st.session_state:
    st.session_state.selected_job_index = None
if 'dashboard_ready' not in st.session_state:
    st.session_state.dashboard_ready = False
if 'user_skills_embeddings_cache' not in st.session_state:
    st.session_state.user_skills_embeddings_cache = {}
if 'skill_embeddings_cache' not in st.session_state:
    st.session_state.skill_embeddings_cache = {}

# WebSocket connection tracking for recovery
if 'ws_connection_time' not in st.session_state:
    st.session_state.ws_connection_time = time.time()
if 'ws_last_activity' not in st.session_state:
    st.session_state.ws_last_activity = time.time()
if 'ws_reconnect_count' not in st.session_state:
    st.session_state.ws_reconnect_count = 0

# Update last activity timestamp on each rerun
st.session_state.ws_last_activity = time.time()

# Check for potential reconnection (session was idle for > 30 seconds)
# This indicates a possible reconnection after connection loss
if time.time() - st.session_state.get('ws_last_activity', time.time()) > 30:
    st.session_state.ws_reconnect_count += 1
    st.session_state.ws_connection_time = time.time()
    # Clear potentially stale data but preserve user input
    if 'matched_jobs' in st.session_state and len(st.session_state.matched_jobs) > 0:
        # Keep matched jobs for continuity but note reconnection
        pass

# Limit search history size
MAX_SEARCH_HISTORY = 20
if len(st.session_state.search_history) > MAX_SEARCH_HISTORY:
    st.session_state.search_history = st.session_state.search_history[-MAX_SEARCH_HISTORY:]

# Run memory cleanup after session state is initialized (if modules available)
if MODULES_AVAILABLE:
    try:
        _cleanup_session_state()
    except Exception:
        pass

# Periodic WebSocket keepalive on app rerun
try:
    _ensure_websocket_alive()
except Exception:
    pass

# APP UI
def main_analyzer_page():
    """Main Page - CareerLens"""
    # Load logo
    logo_img = ""
    try:
        with open("Logo.jpg", "rb") as f:
            logo_base64 = base64.b64encode(f.read()).decode()
            logo_img = f'<img src="data:image/jpeg;base64,{logo_base64}" style="width: 100%; max-width: 400px; margin: 0 auto 2rem auto; display: block;">'
    except (FileNotFoundError, IOError, OSError) as e:
        # Logo file not found or unreadable - continue without logo
        pass
    
    st.markdown(f"""
    <style>
        .main-title {{
            font-family: 'Montserrat', sans-serif;
            font-size: 3rem;
            font-weight: 700;
            letter-spacing: -1px;
            text-align: center;
            margin-bottom: 1rem;
        }}
        .main-title .brand-span {{
            color: var(--brand-core);
        }}
        .main-title .lens-span {{
            color: var(--brand-glow);
        }}
        .main-tagline {{
            text-align: center;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 2px;
            font-size: 0.9rem;
            margin-bottom: 2rem;
        }}
    </style>
    {logo_img}
    """, unsafe_allow_html=True)
    st.markdown("Upload your CV and let **GPT-4** find matching jobs globally, ranked by match quality!")

    # Define helper functions
    def smart_select_match(value, options):
        """Smart match select box options"""
        if not value:
            return 0
        
        value_str = str(value).lower()
        for i, option in enumerate(options):
            if option.lower() in value_str or value_str in option.lower():
                return i
        return 0

    def format_ai_data(data, default=""):
        """Format AI returned data"""
        if isinstance(data, list):
            return ", ".join(data)
        elif isinstance(data, str):
            return data
        else:
            return default


    # Main Page - CV Upload Section
    st.header("📁 Upload Your CV")
    cv_file = st.file_uploader("Choose your CV", type=['pdf', 'docx'], key="cv_uploader")

    # Initialize variables
    autofill_data = {}
    analysis_complete = False
    ai_analysis = {}  # Initialize ai_analysis

    if cv_file:
        st.success(f"✅ Uploaded: **{cv_file.name}**")

        if st.button("🔍 Analyze with GPT-4", type="primary", use_container_width=True, key="analyze_button"):

            # STEP 1: Analyze Resume with WebSocket keepalive
            with ProgressTracker("Analyzing your resume with GPT-4", total_steps=3) as tracker:
                try:
                    tracker.update(1, "Extracting text from your CV...")
                    _websocket_keepalive("Extracting text")
                    
                    tracker.update(2, "AI is analyzing your skills and experience...")
                    _websocket_keepalive("AI analysis in progress")
                    
                    resume_data, ai_analysis = backend.process_resume(cv_file, cv_file.name)
                    
                    tracker.update(3, "Analysis complete!")
                    _websocket_keepalive("Complete")
                    
                    # Check if analysis failed or was incomplete
                    analysis_failed = ai_analysis.get('_analysis_failed', False)
                    analysis_incomplete = ai_analysis.get('_analysis_incomplete', False)
                    
                    if analysis_failed:
                        error_msg = ai_analysis.get('_error', 'Unknown error')
                        st.error(
                            "❌ **AI Analysis Failed**\n\n"
                            f"**Error:** {error_msg}\n\n"
                        )
                        
                        # Show helpful instructions if it's a configuration error
                        if "Missing required secrets" in error_msg:
                            st.info(
                                "💡 **How to fix this:**\n\n"
                                "1. **In Streamlit Cloud:** \n"
                                "   - Go to your app dashboard\n"
                                "   - Click on **⚙️ Settings** → **Secrets**\n"
                                "   - Add the missing API keys\n\n"
                                "2. **For local development:**\n"
                                "   - Create a file `.streamlit/secrets.toml` in your project\n"
                                "   - Add your API keys in TOML format\n\n"
                                "3. **Get API keys:**\n"
                                "   - Azure OpenAI: https://azure.microsoft.com/en-us/products/ai-services/openai-service\n"
                                "   - RapidAPI: https://rapidapi.com/\n\n"
                                "After adding secrets, refresh the page."
                            )
                        
                        st.warning(
                            "⚠️ Please fill in your career details manually in the form below."
                        )
                    elif analysis_incomplete:
                        st.warning(
                            "⚠️ **AI Analysis Incomplete**\n\n"
                            "Could not fully analyze your resume. Please review and complete the form below."
                        )
                    else:
                        st.balloons()

                    # Display analysis results
                    st.markdown("---")
                    st.subheader("🤖 GPT-4 Career Analysis")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        primary_role = ai_analysis.get('primary_role', '') or 'Not detected'
                        st.metric("🎯 Primary Role", primary_role)

                    with col2:
                        confidence = ai_analysis.get('confidence', 0) * 100
                        st.metric("💯 Confidence", f"{confidence:.0f}%")

                    with col3:
                        st.metric("📊 Seniority", ai_analysis.get('seniority_level', '') or 'Not detected')

                    # Skills detected by GPT-4
                    st.markdown("### 💡 Skills Detected by GPT-4")
                    skills = ai_analysis.get('skills', [])
                    if skills:
                        # Create skill tags
                        skills_html = ""
                        for skill in skills[:10]:
                            skills_html += f'<span style="background-color: #E8F4FD; padding: 5px 10px; margin: 3px; border-radius: 5px; display: inline-block;">{skill}</span> '
                        st.markdown(skills_html, unsafe_allow_html=True)

                        if len(skills) > 10:
                            with st.expander(f"➕ Show all {len(skills)} skills"):
                                more_skills_html = ""
                                for skill in skills[10:]:
                                    more_skills_html += f'<span style="background-color: #F0F0F0; padding: 5px 10px; margin: 3px; border-radius: 5px; display: inline-block;">{skill}</span> '
                                st.markdown(more_skills_html, unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ No skills detected")

                    # Core strengths
                    st.markdown("### 💪 Core Strengths")
                    strengths = ai_analysis.get('core_strengths', [])
                    if strengths:
                        cols = st.columns(min(3, len(strengths)))
                        for i, strength in enumerate(strengths):
                            with cols[i % len(cols)]:
                                st.info(f"✓ {strength}")

                    # Extract and format data
                    autofill_data = {
                        # Educational background
                        "education_level": format_ai_data(ai_analysis.get('education_level', '')),
                        "major": format_ai_data(ai_analysis.get('major', '')),
                        "graduation_status": format_ai_data(ai_analysis.get('graduation_status', '')),
                        "university_background": format_ai_data(ai_analysis.get('university_background', '')),
                        
                        # Languages and certificates
                        "languages": format_ai_data(ai_analysis.get('languages', '')),
                        "certificates": format_ai_data(ai_analysis.get('certificates', '')),
                        
                        # Skills - directly use detected skills
                        "hard_skills": format_ai_data(skills),  # Use detected skills
                        "soft_skills": format_ai_data(ai_analysis.get('core_strengths', [])),  # Use core strengths
                        
                        # Work experience
                        "work_experience": format_ai_data(ai_analysis.get('work_experience', '')),
                        "project_experience": format_ai_data(ai_analysis.get('project_experience', '')),
                        
                        # Preferences
                        "location_preference": format_ai_data(ai_analysis.get('location_preference', '')),
                        "industry_preference": format_ai_data(ai_analysis.get('industry_preference', '')),
                        
                        # Salary
                        "salary_expectation": format_ai_data(ai_analysis.get('salary_expectation', '')),
                        "benefits_expectation": format_ai_data(ai_analysis.get('benefits_expectation', '')),
                        
                        # New fields
                        "primary_role": format_ai_data(ai_analysis.get('primary_role', '')),
                        "simple_search_terms": format_ai_data(ai_analysis.get('simple_search_terms', ''))
                    }
                    
                    analysis_complete = True
                    
                    # Store in session state
                    st.session_state.autofill_data = autofill_data
                    st.session_state.analysis_complete = True
                    st.session_state.ai_analysis = ai_analysis  # Save ai_analysis for later use

                    st.success("🎉 Resume analysis complete! Form has been auto-filled with your information.")

                except Exception as e:
                    st.error(f"❌ Error analyzing resume: {str(e)}")
                    st.stop()

    else:
        # Welcome screen
        st.info("📄 **Upload your CV above to get started!**")

        # Instructions
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### 📋 How it works:

            1. **📄 Upload** your CV (PDF or DOCX)
            2. **🤖 GPT-4** analyzes your skills, experience, and ideal roles
            3. **🔍 Search** LinkedIn jobs via RapidAPI (global search)
            4. **🎯 Rank** all jobs by match quality using AI
            5. **📊 See** your best matches with detailed scores!
            """)

        st.markdown("---")
        st.success("💡 **Pro Tip:** Jobs are searched globally (not filtered by Hong Kong) and ranked by how well they match your profile!")

    # ========== Form Area ==========
    if st.session_state.get('analysis_complete', False) or not cv_file:
        with st.form("job_seeker_form"):
            st.subheader("📝 Complete Your Profile")
            
            if st.session_state.get('analysis_complete', False):
                st.success("✅ Form auto-filled with your resume analysis!")
            
            st.markdown("Review and edit the auto-filled information from your CV analysis:")

            # Use data from session_state
            current_data = st.session_state.get('autofill_data', {})

            # Career Preferences - new fields at top of form
            st.subheader("🎯 Career Preferences")
            col_career1, col_career2 = st.columns(2)
            
            with col_career1:
                primary_role = st.text_input("Primary Role*", 
                                           value=current_data.get("primary_role", ""),
                                           placeholder="e.g., Project Manager, Software Engineer, Data Analyst")
            
            with col_career2:
                simple_search_terms = st.text_input("Search Keywords*", 
                                                  value=current_data.get("simple_search_terms", ""),
                                                  placeholder="e.g., python developer, project management, data science")

            # Educational background
            st.subheader("🎓 Educational background")
            col1, col2 = st.columns(2)

            with col1:
                education_options = ["Please select", "PhD", "Master", "Bachelor", "Diploma", "High School"]
                ed_level = current_data.get("education_level", "")
                education_index = smart_select_match(ed_level, education_options)
                
                education_level = st.selectbox(
                    "Educational level*",
                    education_options,
                    index=education_index
                )
                
                major = st.text_input("Major", 
                                    value=current_data.get("major", ""),
                                    placeholder="e.g., Computer Science, Business Administration")
                
                grad_options = ["Please select", "Graduated", "Fresh graduates", "Currently studying"]
                grad_status = current_data.get("graduation_status", "")
                grad_index = smart_select_match(grad_status, grad_options)
                
                graduation_status = st.selectbox(
                    "Graduation status*",
                    grad_options,
                    index=grad_index
                )

            with col2:
                uni_options = ["Please select", "985 Universities", "211 Universities", "Overseas Universities", "Regular Undergraduate Universities", "Other"]
                uni_bg = current_data.get("university_background", "")
                uni_index = smart_select_match(uni_bg, uni_options)
                
                university_background = st.selectbox(
                    "University background*",
                    uni_options,
                    index=uni_index
                )
                
                languages = st.text_input("Languages", 
                                        value=current_data.get("languages", ""),
                                        placeholder="e.g., English, Mandarin, Cantonese")
                
                certificates = st.text_input("Certificates", 
                                           value=current_data.get("certificates", ""),
                                           placeholder="e.g., PMP, CFA, AWS Certified")

            # Skills
            st.subheader("💼 Skills")
            hard_skills = st.text_area("Technical Skills", 
                                     value=current_data.get("hard_skills", ""),
                                     placeholder="e.g., Python, JavaScript, SQL, Machine Learning",
                                     height=100)
            
            soft_skills = st.text_area("Core Strengths", 
                                     value=current_data.get("soft_skills", ""),
                                     placeholder="e.g., Leadership, Communication, Problem Solving",
                                     height=100)

            # Work Experience
            st.subheader("📈 Work Experience")
            col3, col4 = st.columns(2)

            with col3:
                work_exp_options = ["Please select", "Recent Graduate", "1-3 years", "3-5 years", "5-10 years", "10+ years"]
                work_exp = current_data.get("work_experience", "")
                work_index = smart_select_match(work_exp, work_exp_options)
                
                work_experience = st.selectbox(
                    "Work experience years*",
                    work_exp_options,
                    index=work_index
                )

            with col4:
                project_experience = st.text_area("Project experience", 
                                                value=current_data.get("project_experience", ""),
                                                placeholder="Describe your key projects and achievements",
                                                height=100)

            # Work preferences
            st.subheader("📍 Work preferences")
            col5, col6 = st.columns(2)

            with col5:
                loc_options = ["Please select", "Hong Kong", "Mainland China", "Overseas", "No Preference"]
                loc_pref = current_data.get("location_preference", "")
                loc_index = smart_select_match(loc_pref, loc_options)
                
                location_preference = st.selectbox(
                    "Location Preference*",
                    loc_options,
                    index=loc_index
                )
             
            with col6:
                industry_preference = st.text_input("Industry Preference", 
                                                  value=current_data.get("industry_preference", ""),
                                                  placeholder="e.g., Technology, Finance, Healthcare")
       
            # Salary and benefits expectations
            st.subheader("💰 Salary and Benefits Expectations")
            salary_expectation = st.text_input("Expected Salary Range", 
                                             value=current_data.get("salary_expectation", ""),
                                             placeholder="e.g., HKD 30,000 - 40,000")
            
            benefits_expectation = st.text_area("Benefits Requirements", 
                                              value=current_data.get("benefits_expectation", ""),
                                              placeholder="e.g., Medical insurance, Flexible working hours",
                                              height=80)
            

            # Submit button
            submitted = st.form_submit_button("💾 Save Information", use_container_width=True)

            if submitted:
                if (education_level == "Please select" or graduation_status == "Please select" or
                    university_background == "Please select" or work_experience == "Please select" or
                    location_preference == "Please select" or not primary_role.strip() or not simple_search_terms.strip()):
                    st.error("Please complete all required fields (marked with *)!")
                else:
                    # Save to database
                    job_seeker_id = save_job_seeker_info(
                        education_level, major, graduation_status, university_background,
                        languages, certificates, hard_skills, soft_skills, work_experience,
                        project_experience, location_preference, industry_preference,
                        salary_expectation, benefits_expectation,
                        primary_role,  # Use value from form
                        simple_search_terms  # Use value from form
                    )
                    
                    if job_seeker_id:
                        # Save to session state
                        st.session_state.job_seeker_id = job_seeker_id
                        st.success(f"✅ Information saved successfully! Your ID: {job_seeker_id}")
                        st.balloons()
                        
                        # Display success message
                        st.info(f"🔑 Your job seeker ID has been saved: **{job_seeker_id}**")
                        st.info("💡 You can use this ID on the Job Match page to view personalized job recommendations")
                    else:
                        st.error("❌ Failed to save information, please try again")

    """Save job seeker information to database"""

def job_recommendations_page(job_seeker_id=None):
    """Job Recommendations Page - Using Real API Data"""
    st.title("💼 Personalized Job Recommendations")

    # Get job seeker data - add error handling
    job_seeker_data = None
    try:
        if job_seeker_id:
            job_seeker_data = db.get_job_seeker_by_id(job_seeker_id)
        else:
            # If no ID provided, try to get latest record
            job_seeker_data = db.get_latest_job_seeker_data()
            
    except Exception as e:
        st.error(f"Error getting job seeker data: {e}")
        return

    if not job_seeker_data:
        st.error("No job seeker information found, please fill in your personal information first")
        st.info("Please fill in your information on the Job Seeker page")
        
        # Display debug information
        with st.expander("🔍 Debug Information"):
            st.write(f"Provided job_seeker_id: {job_seeker_id}")
            st.write("Trying to get latest record...")
            latest_id = db.get_latest_job_seeker_id()
            st.write(f"Latest record ID: {latest_id}")
            
        return

    # Display personal information summary
    with st.expander("👤 Your Personal Information"):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Education:** {job_seeker_data.get('education_level', 'N/A')}")
            st.write(f"**Major:** {job_seeker_data.get('major', 'N/A')}")
            st.write(f"**Experience:** {job_seeker_data.get('work_experience', 'N/A')}")
            st.write(f"**Primary Role:** {job_seeker_data.get('primary_role', 'N/A')}")
        with col2:
            st.write(f"**Location Preference:** {job_seeker_data.get('location_preference', 'N/A')}")
            st.write(f"**Industry Preference:** {job_seeker_data.get('industry_preference', 'N/A')}")
            st.write(f"**Search Keywords:** {job_seeker_data.get('simple_search_terms', 'N/A')}")

    # Display skill information
    with st.expander("💼 Skill Information"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Technical Skills:**")
            hard_skills = job_seeker_data.get('hard_skills', '')
            if hard_skills:
                skills_list = [skill.strip() for skill in hard_skills.split(',')]
                for skill in skills_list[:10]:  # Show first 10 skills
                    st.write(f"• {skill}")
        with col2:
            st.write("**Core Strengths:**")
            soft_skills = job_seeker_data.get('soft_skills', '')
            if soft_skills:
                strengths_list = [strength.strip() for strength in soft_skills.split(',')]
                for strength in strengths_list[:5]:  # Show first 5 core strengths
                    st.write(f"• {strength}")

    # ----------------------------------------
    # 🔍 Job Search Settings
    # ----------------------------------------
    st.subheader("🔍 Job Search Settings")

    # Pre-fill defaults using job seeker data
    default_search = (
        job_seeker_data.get("primary_role", "")
        or job_seeker_data.get("simple_search_terms", "Python developer")
    )

    default_location = job_seeker_data.get("location_preference", "Hong Kong")

    col1, col2, col3 = st.columns(3)

    with col1:
        search_query = st.text_input(
            "Job Keywords*",
            value=default_search,
            placeholder="e.g.: software engineer, data analyst"
        )

    with col2:
        location = st.text_input(
            "City/Region",
            value=default_location,
            placeholder="e.g.: New York, London"
        )

    with col3:
        country = st.selectbox(
            "Country Code",
            ["hk", "us", "gb", "ca", "au", "sg"],
            index=0
        )

    col4, = st.columns(1)

    with col4:
        employment_types = st.multiselect(
            "Employment Type",
            ["FULLTIME", "PARTTIME", "CONTRACTOR"],
            default=["FULLTIME"]
        )


    # ----------------------------------------
    # 🔧 Search Speed Options
    # ----------------------------------------
    st.markdown("##### ⚡ Search Mode")
    
    search_mode = st.radio(
        "Choose search speed:",
        ["⚡ Quick Search (15 jobs)", "🔍 Standard Search (25 jobs)", "🔬 Deep Search (40 jobs)"],
        index=0,
        horizontal=True,
        help="Quick = faster results, Deep = more comprehensive but slower"
    )
    
    # Map search mode to job count
    search_mode_map = {
        "⚡ Quick Search (15 jobs)": 15,
        "🔍 Standard Search (25 jobs)": 25,
        "🔬 Deep Search (40 jobs)": 40
    }
    num_jobs_to_search = search_mode_map.get(search_mode, 15)
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_jobs_to_show = st.slider(
            "Top matches to display", 
            3, 15, 5,
            key="jobs_show_slider"
        )
    
    with col2:
        # Show estimated time based on search mode
        time_estimates = {
            "⚡ Quick Search (15 jobs)": "~30-60 seconds",
            "🔍 Standard Search (25 jobs)": "~60-90 seconds",
            "🔬 Deep Search (40 jobs)": "~90-120 seconds"
        }
        st.info(f"⏱️ Estimated time: {time_estimates.get(search_mode, '~60 seconds')}")
    # -------------------------------------------------------
    # 🔎 STEP 2: Search Jobs via RapidAPI (SAFE VERSION)
    # -------------------------------------------------------
    # Use ProgressTracker to maintain WebSocket connection during long operations
    with ProgressTracker(f"Searching {num_jobs_to_search} jobs", total_steps=5) as tracker:

        try:
            # ----------------------------------------------------
            # 1) Load job seeker ID safely
            # ----------------------------------------------------
            tracker.update(1, "Loading profile settings...")
            _websocket_keepalive("Loading profile")
            
            current_id = st.session_state.get("job_seeker_id")

            if not current_id:
                st.warning("⚠ job_seeker_id not found in session — using default search settings.")
                search_fields = {
                    "primary_role": "",
                    "simple_search_terms": "",
                    "location_preference": "Hong Kong",
                    "hard_skills": ""
                }
            else:
                # ----------------------------------------------------
                # 2) Load DB Search Fields
                # ----------------------------------------------------
                try:
                    search_fields = get_job_seeker_search_fields(current_id)
                except Exception as db_err:
                    st.error(f"❌ Database error when loading search settings: {db_err}")
                    search_fields = None

                if not search_fields:
                    st.warning("⚠ No stored search preferences found — using default search settings.")
                    search_fields = {
                        "primary_role": "",
                        "simple_search_terms": "",
                        "location_preference": "Hong Kong",
                        "hard_skills": ""
                    }

            # Extract fields
            primary_role        = search_fields.get("primary_role", "")
            simple_search_terms = search_fields.get("simple_search_terms", "")
            location_preference = search_fields.get("location_preference", "Hong Kong")
            hard_skills         = search_fields.get("hard_skills", "")

            tracker.update(2, "Preparing search query...")
            _websocket_keepalive("Preparing query")
            
            # Construct resume_data with all fields
                        
            resume_data = {
                "education_level": job_seeker_data.get("education_level", ""),
                "major": job_seeker_data.get("major", ""),
                "graduation_status": job_seeker_data.get("graduation_status", ""),
                "university_background": job_seeker_data.get("university_background", ""),
                "languages": job_seeker_data.get("languages", ""),
                "certificates": job_seeker_data.get("certificates", ""),
                "hard_skills": job_seeker_data.get("hard_skills", ""),
                "soft_skills": job_seeker_data.get("soft_skills", ""),
                "work_experience": job_seeker_data.get("work_experience", ""),
                "project_experience": job_seeker_data.get("project_experience", ""),
                "location_preference": job_seeker_data.get("location_preference", ""),
                "industry_preference": job_seeker_data.get("industry_preference", ""),
                "salary_expectation": job_seeker_data.get("salary_expectation", ""),
                "benefits_expectation": job_seeker_data.get("benefits_expectation", ""),
                "primary_role": job_seeker_data.get("primary_role", ""),
                "simple_search_terms": job_seeker_data.get("simple_search_terms", ""),
            }

            # Construct ai_analysis dict, which can focus on skills, role, location, etc.
            ai_analysis = {
                "education_level": resume_data["education_level"],
                "major": resume_data["major"],
                "graduation_status": resume_data["graduation_status"],
                "university_background": resume_data["university_background"],
                "languages": [lang.strip() for lang in resume_data["languages"].split(",")] if resume_data["languages"] else [],
                "certificates": [cert.strip() for cert in resume_data["certificates"].split(",")] if resume_data["certificates"] else [],
                "skills": [skill.strip() for skill in resume_data["hard_skills"].split(",")] if resume_data["hard_skills"] else [],
                "soft_skills": [skill.strip() for skill in resume_data["soft_skills"].split(",")] if resume_data["soft_skills"] else [],
                "work_experience": resume_data["work_experience"],
                "project_experience": resume_data["project_experience"],
                "location_preference": resume_data["location_preference"],
                "industry_preference": resume_data["industry_preference"],
                "salary_expectation": resume_data["salary_expectation"],
                "benefits_expectation": resume_data["benefits_expectation"],
                "primary_role": resume_data["primary_role"],
                "simple_search_terms": resume_data["simple_search_terms"],
            }
    
            # ----------------------------------------------------
            # 3) Build search keyword string
            # ----------------------------------------------------
            search_keywords = ", ".join(
                field for field in [
                    primary_role,
                    simple_search_terms,
                    hard_skills, 
                ] if field.strip()
            )

            if not search_keywords:
                search_keywords = "General"

            # ----------------------------------------------------
            # 4) Show user what we are searching
            # ----------------------------------------------------
            st.info(
                f"📡 Searching jobs via RapidAPI:\n\n"
                f"**Keywords:** {search_keywords}\n"
                f"**Location:** {location_preference}"
            )

            tracker.update(3, "Connecting to job search API...")
            _websocket_keepalive("Connecting to API")

            # ----------------------------------------------------
            # 5) Search and Match Jobs via Backend (SINGLE search - no duplicates)
            # ----------------------------------------------------
            tracker.update(4, f"Searching & matching {num_jobs_to_search} jobs...")
            _websocket_keepalive("Searching jobs")

            matched_jobs = backend.search_and_match_jobs(
                resume_data=resume_data,
                ai_analysis=ai_analysis,
                num_jobs=num_jobs_to_search,
                search_keywords=search_keywords,
                location=location_preference
            )
            
            tracker.update(5, "Search complete!")
            _websocket_keepalive("Complete")
            
            # Better error feedback for users
            if not matched_jobs:
                st.warning(
                    "⚠️ **No jobs found.** This could be due to:\n\n"
                    "• **Rate limit**: RapidAPI free tier has limited requests. Wait a few minutes.\n"
                    "• **Search terms**: Try broader keywords in your profile.\n"
                    "• **Location**: Try a different location preference.\n"
                    "• **API issues**: Check if your RapidAPI key is valid."
                )
        except Exception as e:
            st.error(
                f"❌ **Search failed:** {str(e)}\n\n"
                "**Troubleshooting:**\n"
                "• Check your internet connection\n"
                "• Verify RapidAPI key is configured correctly\n"
                "• Try again in a few minutes if rate limited"
            )
            matched_jobs = []

        # ----------------------------------------
        # 📊 STEP 3: Display Results
        # ----------------------------------------
        st.markdown("---")

        if matched_jobs and len(matched_jobs) > 0:

            st.success(f"✅ Step 3/3: Found & ranked **{len(matched_jobs)}** jobs by match quality!")
            st.markdown(f"## 🎯 Top {num_jobs_to_show} Job Matches")

            st.info("📊 **Ranking Algorithm:** 60% Semantic Similarity + 40% Skill Match")

            # Display top matches
            for i, job in enumerate(matched_jobs[:num_jobs_to_show], start=1):

                combined = job.get("combined_score", 0)

                if combined >= 80:
                    match_emoji, match_label, match_color = "🟢", "Excellent Match", "#D4EDDA"
                elif combined >= 60:
                    match_emoji, match_label, match_color = "🟡", "Good Match", "#FFF3CD"
                else:
                    match_emoji, match_label, match_color = "🟠", "Fair Match", "#F8D7DA"

                expander_title = (
                    f"**#{i}** • {job.get('title', 'Unknown')} at {job.get('company', 'Unknown')} "
                    f"- {match_emoji} {match_label} ({combined:.1f}%)"
                )

                with st.expander(expander_title, expanded=i <= 2):

                    # Scores
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🎯 Combined Score", f"{combined:.1f}%")
                    with col2:
                        st.metric("🧠 Semantic Match", f"{job.get('semantic_score', 0):.1f}%")
                    with col3:
                        st.metric("✅ Skill Match", f"{job.get('skill_match_percentage', 0):.1f}%")
                    with col4:
                        st.metric("🔢 Skills Matched", job.get("matched_skills_count", 0))

                    # Job details
                    st.markdown("##### 📋 Job Details")
                    detail_col1, detail_col2 = st.columns(2)

                    with detail_col1:
                        st.write(f"**📍 Location:** {job.get('location', 'Unknown')}")
                        st.write(f"**🏢 Company:** {job.get('company', 'Unknown')}")

                    with detail_col2:
                        st.write(f"**📅 Posted:** {job.get('posted_date', 'Unknown')}")
                        st.write(f"**💼 Role:** {job.get('title', 'Unknown')}")

                    # Matched skills (candidate has)
                    matched_skills = job.get("matched_skills", [])

                    # Required skills from job (assumes this field exists as a list)
                    required_skills = job.get("required_skills", [])

                    # Skills to improve: required but NOT matched
                    skills_to_improve = []
                    if required_skills:
                        required_set = set([s.lower() for s in required_skills])
                        matched_set = set([s.lower() for s in matched_skills])
                        missing_skills = required_set - matched_set
                        skills_to_improve = list(missing_skills)

                    # Display matched skills section
                    if matched_skills:
                        st.markdown("##### ✨ Your Skills That Match This Job")

                        badge_html = "".join(
                            f"""
                            <span style="
                                background-color:#D4EDDA;
                                color:#155724;
                                padding:5px 10px;
                                margin:3px;
                                border-radius:5px;
                                display:inline-block;
                                font-weight:bold;
                            ">✓ {skill}</span>
                            """
                            for skill in matched_skills[:8]
                        )

                        st.markdown(badge_html, unsafe_allow_html=True)

                        if len(matched_skills) > 8:
                            st.caption(f"+ {len(matched_skills) - 8} more matching skills")

                    # Display skills to improve section
                    if skills_to_improve:
                        st.markdown("##### 🛠 Skills You May Want to Improve")

                        badge_html_improve = "".join(
                            f"""
                            <span style="
                                background-color:#F8D7DA;
                                color:#721C24;
                                padding:5px 10px;
                                margin:3px;
                                border-radius:5px;
                                display:inline-block;
                                font-weight:bold;
                            ">✗ {skill}</span>
                            """
                            for skill in skills_to_improve[:8]
                        )

                        st.markdown(badge_html_improve, unsafe_allow_html=True)

                        if len(skills_to_improve) > 8:
                            st.caption(f"+ {len(skills_to_improve) - 8} more skills to consider")

                    # Description
                    description = job.get("description", "")
                    if description:
                        st.markdown("##### 📝 Job Description")
                        preview = description[:500]
                        st.text_area(
                            "Preview",
                            preview + ("..." if len(description) > 500 else ""),
                            height=120,
                            key=f"desc_{job.get('id', i)}"
                        )

                    # Action buttons
                    col_btn1, col_btn2 = st.columns(2)
                    
                    with col_btn1:
                        # Apply link
                        job_url = job.get("url", "")
                        if job_url:
                            st.link_button(
                                "🔗 Apply Now",
                                job_url,
                                use_container_width=True,
                                type="primary"
                            )
                        else:
                            st.info("🔗 Link not available")
                    
                    with col_btn2:
                        # Resume tailoring button
                        if st.button("✨ Tailor Resume", key=f"tailor_{job.get('id', i)}", use_container_width=True):
                            st.session_state.selected_job_for_resume = job
                            st.session_state.show_resume_generator = True
                
                # Show resume generator if selected
                if st.session_state.get('show_resume_generator') and st.session_state.get('selected_job_for_resume', {}).get('id') == job.get('id'):
                    with st.container():
                        st.markdown("---")
                        display_resume_generator_ui(
                            job, 
                            job_seeker_data,
                            resume_text=st.session_state.get('resume_text')
                        )
                        if st.button("❌ Close Resume Generator", key=f"close_resume_{job.get('id', i)}"):
                            st.session_state.show_resume_generator = False
                            st.session_state.selected_job_for_resume = None
                            st.rerun()

        else:
            st.warning("⚠️ No matched jobs found. Please try adjusting your search criteria.")
            
    if matched_jobs and len(matched_jobs) > 0:
        # Display CareerLens Market Positioning Dashboard
        display_market_positioning_dashboard(matched_jobs, job_seeker_data)
        
        # Create enhanced visualizations
        create_enhanced_visualizations(matched_jobs, job_seeker_data)
        
        # Create radar chart comparison for top jobs
        create_job_comparison_radar(matched_jobs)
        
        # Additional detailed analysis
        st.markdown("---")
        st.subheader("🔍 Deep Dive Analysis")
        
        # Industry distribution of matched jobs
        industries = {}
        for job in matched_jobs:
            # Extract industry from company or description
            company = job.get('company', '').lower()
            if any(tech in company for tech in ['tech', 'software', 'ai', 'data']):
                industry = 'Technology'
            elif any(finance in company for finance in ['bank', 'finance', 'investment', 'capital']):
                industry = 'Finance'
            elif any(consult in company for consult in ['consulting', 'consultancy']):
                industry = 'Consulting'
            else:
                industry = 'Other'
            
            industries[industry] = industries.get(industry, 0) + 1
        
        if industries:
            st.markdown("#### 🏭 Industries in Your Matches")
            for industry, count in industries.items():
                percentage = (count / len(matched_jobs)) * 100
                st.write(f"- **{industry}**: {count} jobs ({percentage:.1f}%)")

    else:
        st.warning("⚠️ No matched jobs found. Please try adjusting your search criteria.")


def enhanced_head_hunter_page():
    """Enhanced Head Hunter Page - Job Publishing and Management"""
    st.title("🎯 Head Hunter Portal")

    # Page selection
    page_option = st.sidebar.radio(
        "Select Function",
        ["Publish New Position", "View Published Positions", "Position Statistics"]
    )

    if page_option == "Publish New Position":
        publish_new_job()
    elif page_option == "View Published Positions":
        view_published_jobs()
    elif page_option == "Position Statistics":
        show_job_statistics()

def publish_new_job():
    """Publish New Position Form"""
    st.header("📝 Publish New Position")

    with st.form("head_hunter_job_form"):
        # Basic Position Information
        st.subheader("🎯 Basic Position Information")

        col1, col2 = st.columns(2)
        with col1:
            job_title = st.text_input("Position Title*", placeholder="e.g.: Senior Frontend Engineer")
        with col2:
            employment_type = st.selectbox("Employment Type*", ["Please select", "Full-time", "Part-time", "Contract", "Internship"])

        job_description = st.text_area("Job Description*", height=100,
                                      placeholder="Detailed introduction of position main content and team situation...")

        main_responsibilities = st.text_area("Main Responsibilities*", height=100,
                                           placeholder="List main responsibilities with bullet points, one per line...")

        required_skills = st.text_area("Required Skills & Qualifications*", height=100,
                                     placeholder="e.g.: 5+ years experience, proficient in React.js, Computer Science degree...")

        # Company and Client Information
        st.subheader("🏢 Company and Client Information")

        col3, col4 = st.columns(2)
        with col3:
            client_company = st.text_input("Client Company Name*", placeholder="Company official name")
            industry = st.selectbox("Industry*", ["Please select", "Technology", "Finance", "Consulting", "Healthcare", "Education", "Manufacturing", "Retail", "Other"])
        with col4:
            work_location = st.selectbox("Work Location*", ["Please select", "Hong Kong", "Mainland China", "Overseas", "Remote"])
            company_size = st.selectbox("Company Size*", ["Please select", "Startup (1-50)", "SME (51-200)", "Large Enterprise (201-1000)", "Multinational (1000+)"])

        work_type = st.selectbox("Work Type*", ["Please select", "Remote", "Hybrid", "Office"])

        # Employment Details
        st.subheader("💼 Employment Details")

        col5, col6 = st.columns(2)
        with col5:
            experience_level = st.selectbox("Experience Level*", ["Please select", "Fresh Graduate", "1-3 years", "3-5 years", "5-10 years", "10+ years"])
        with col6:
            visa_support = st.selectbox("Visa Support", ["Not provided", "Work Visa", "Assistance provided", "Must have own visa"])

        # Salary and Application Method
        st.subheader("💰 Salary and Application Method")

        col7, col8, col9 = st.columns([2, 2, 1])
        with col7:
            min_salary = st.number_input("Minimum Salary*", min_value=0, value=30000, step=5000)
        with col8:
            max_salary = st.number_input("Maximum Salary*", min_value=0, value=50000, step=5000)
        with col9:
            currency = st.selectbox("Currency", ["HKD", "USD", "CNY", "EUR", "GBP"])

        benefits = st.text_area("Benefits", height=80,
                              placeholder="e.g.: Medical insurance, 15 days annual leave, performance bonus, stock options...")

        application_method = st.text_area("Application Method*", height=80,
                                        value="Please send resume to recruit@headhunter.com, include position title in email subject",
                                        placeholder="Application process and contact information...")

        # Use timedelta from datetime instead of pd.Timedelta for faster loading
        from datetime import timedelta
        job_valid_until = st.date_input("Position Posting Validity Period*",
                                      value=datetime.now().date() + timedelta(days=30))

        # Submit button
        submitted = st.form_submit_button("💾 Publish Position", type="primary", use_container_width=True)

        if submitted:
            # Validate required fields
            required_fields = [
                job_title, job_description, main_responsibilities, required_skills,
                client_company, industry, work_location, work_type, company_size,
                employment_type, experience_level, min_salary, max_salary, application_method
            ]

            if "Please select" in [employment_type, industry, work_location, work_type, company_size, experience_level]:
                st.error("Please complete all required fields (marked with *)!")
            elif not all(required_fields):
                st.error("Please complete all required fields (marked with *)!")
            elif min_salary >= max_salary:
                st.error("Maximum salary must be greater than minimum salary!")
            
            # Modify this part in Streamlit app:
            else:
                # Create dictionary object
                job_data = {
                    'job_title': job_title,
                    'job_description': job_description,
                    'main_responsibilities': main_responsibilities,
                    'required_skills': required_skills,
                    'client_company': client_company,
                    'industry': industry,
                    'work_location': work_location,
                    'work_type': work_type,
                    'company_size': company_size,
                    'employment_type': employment_type,
                    'experience_level': experience_level,
                    'visa_support': visa_support,
                    'min_salary': min_salary,
                    'max_salary': max_salary,
                    'currency': currency,
                    'benefits': benefits,
                    'application_method': application_method,
                    'job_valid_until': job_valid_until.strftime("%Y-%m-%d")
                }
                
                # Save to database - now only pass one parameter
                success = save_head_hunter_job(job_data)

                if success:
                    st.success("✅ Position published successfully!")
                    st.balloons()
                else:
                    st.error("❌ Position publishing failed, please try again")


def view_published_jobs():
    """View Published Positions"""
    st.header("📋 Published Positions")

    jobs = db2.get_all_head_hunter_jobs()

    if not jobs:
        st.info("No positions published yet")
        return

    st.success(f"Published {len(jobs)} positions")

    # Search and filter
    col1, col2 = st.columns(2)
    with col1:
        search_term = st.text_input("Search position title or company")
    with col2:
        filter_industry = st.selectbox("Filter by industry", ["All industries"] + ["Technology", "Finance", "Consulting", "Healthcare", "Education", "Manufacturing", "Retail", "Other"])

    # Filter positions
    filtered_jobs = jobs
    if search_term:
        filtered_jobs = [job for job in jobs if search_term.lower() in job[2].lower() or search_term.lower() in job[6].lower()]
    if filter_industry != "All industries":
        filtered_jobs = [job for job in filtered_jobs if job[7] == filter_industry]

    if not filtered_jobs:
        st.warning("No matching positions found")
        return

    # Display position list
    for job in filtered_jobs:
        with st.expander(f"#{job[0]} {job[2]} - {job[6]}", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Published Time:** {job[1]}")
                st.write(f"**Company:** {job[6]}")
                st.write(f"**Industry:** {job[7]}")
                st.write(f"**Location:** {job[8]} ({job[9]})")
                st.write(f"**Size:** {job[10]}")

            with col2:
                st.write(f"**Type:** {job[11]}")
                st.write(f"**Experience:** {job[12]}")
                st.write(f"**Salary:** {job[14]:,} - {job[15]:,} {job[16]}")
                st.write(f"**Valid Until:** {job[19]}")
                if job[13] != "Not provided":
                    st.write(f"**Visa:** {job[13]}")

            st.write("**Description:**")
            st.write(job[3][:200] + "..." if len(job[3]) > 200 else job[3])

def show_job_statistics():
    """Display Position Statistics"""
    st.header("📊 Position Statistics")

    jobs = db2.get_all_head_hunter_jobs()

    if not jobs:
        st.info("No statistics available yet")
        return

    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Positions", len(jobs))
    with col2:
        active_jobs = len([job for job in jobs if datetime.strptime(job[19], "%Y-%m-%d").date() >= datetime.now().date()])
        st.metric("Active Positions", active_jobs)
    with col3:
        expired_jobs = len(jobs) - active_jobs
        st.metric("Expired Positions", expired_jobs)
    with col4:
        avg_salary = sum((job[14] + job[15]) / 2 for job in jobs) / len(jobs)
        st.metric("Average Salary", f"{avg_salary:,.0f}")

    # Industry distribution
    st.subheader("🏭 Industry Distribution")
    industry_counts = {}
    for job in jobs:
        industry = job[7]
        industry_counts[industry] = industry_counts.get(industry, 0) + 1

    for industry, count in industry_counts.items():
        st.write(f"• **{industry}:** {count} positions ({count/len(jobs)*100:.1f}%)")

    # Location distribution
    st.subheader("📍 Work Location Distribution")
    location_counts = {}
    for job in jobs:
        location = job[8]
        location_counts[location] = location_counts.get(location, 0) + 1

    for location, count in location_counts.items():
        st.write(f"• **{location}:** {count} positions")

    # Experience requirement distribution
    st.subheader("🎯 Experience Requirement Distribution")
    experience_counts = {}
    for job in jobs:
        experience = job[12]
        experience_counts[experience] = experience_counts.get(experience, 0) + 1

    for experience, count in experience_counts.items():
        st.write(f"• **{experience}:** {count} positions")

def recruitment_match_dashboard():
    """Recruitment Match Dashboard"""
    st.title("🎯 Recruitment Match Portal")

    # Quick statistics
    jobs = get_all_jobs_for_matching()
    seekers = get_all_job_seekers()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active Positions", len(jobs) if jobs else 0)
    with col2:
        st.metric("Job Seekers", len(seekers) if seekers else 0)
    with col3:
        st.metric("Match Ready", "✅" if jobs and seekers else "❌")

    # Page selection
    page_option = st.sidebar.radio(
        "Select Function",
        ["Smart Talent Matching", "Match Statistics", "Instructions"]
    )

    if page_option == "Smart Talent Matching":
        recruitment_match_page()
    elif page_option == "Match Statistics":
        show_match_statistics()
    else:
        show_instructions()

def recruitment_match_page():
    """Recruitment Match Page"""
    st.title("🎯 Recruitment Match - Smart Talent Matching")

    # Get data
    jobs = get_all_jobs_for_matching()
    seekers = get_all_job_seekers()

    if not jobs:
        st.warning("❌ No available position information, please first publish positions in the headhunter module")
        return

    if not seekers:
        st.warning("❌ No available job seeker information, please first fill in information on Job Seeker page")
        return

    st.success(f"📊 System has {len(jobs)} active positions and {len(seekers)} job seekers")

    # Select position for matching
    st.subheader("🔍 Select Position to Match")

    job_options = {f"#{job[0]} {job[1]} - {job[5]}": job for job in jobs}
    selected_job_key = st.selectbox("Select Position", list(job_options.keys()))
    selected_job = job_options[selected_job_key]

    # Display position details
    with st.expander("📋 Position Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Position ID:** #{selected_job[0]}")
            st.write(f"**Company:** {selected_job[5]}")
            st.write(f"**Industry:** {selected_job[6]}")
            st.write(f"**Experience Requirement:** {selected_job[11]}")
        with col2:
            st.write(f"**Location:** {selected_job[7]}")
            st.write(f"**Salary:** {selected_job[13]:,}-{selected_job[14]:,} {selected_job[15]}")
            st.write(f"**Skill Requirements:** {selected_job[4][:100]}...")

    # Match options
    st.subheader("⚙️ Match Settings")
    col1, col2 = st.columns(2)
    with col1:
        min_match_score = st.slider("Minimum Match Score", 0, 100, 60)
    with col2:
        max_candidates = st.slider("Display Top N Candidates", 1, 20, 10)

    # Execute matching
    if st.button("🚀 Start Smart Matching", type="primary", use_container_width=True):
        st.subheader("📈 Match Results")

        results = []
        total_candidates = min(len(seekers), max_candidates)
        
        # Use ProgressTracker for WebSocket keepalive during matching
        with ProgressTracker("Smart Matching", total_steps=total_candidates) as tracker:
            for i, seeker in enumerate(seekers[:max_candidates]):
                tracker.update(i + 1, f"Analyzing candidate {i + 1}/{total_candidates}...")
                
                # Send keepalive every 3 candidates to maintain connection
                if i % 3 == 0:
                    _websocket_keepalive(f"Matching candidate {i + 1}")

                # Use simplified matching algorithm
                analysis_result = analyze_match_simple(selected_job, seeker)
                match_score = analysis_result.get('match_score', 0)

                if match_score >= min_match_score:
                    results.append({
                        'seeker_id': seeker[0],
                        'name': seeker[1],
                        'current_title': seeker[9],
                        'experience': seeker[3],
                        'education': seeker[4],
                        'match_score': match_score,
                        'analysis': analysis_result,
                        'raw_data': seeker
                    })

        # Display results
        if results:
            results.sort(key=lambda x: x['match_score'], reverse=True)
            st.success(f"🎉 Found {len(results)} matching candidates (score ≥ {min_match_score})")

            for i, result in enumerate(results):
                score_color = "🟢" if result['match_score'] >= 80 else "🟡" if result['match_score'] >= 60 else "🔴"

                with st.expander(f"{score_color} #{i+1} {result['name']} - {result['match_score']} points", expanded=i < 2):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Candidate Information:**")
                        st.write(f"**ID:** #{result['seeker_id']}")
                        st.write(f"**Education Background:** {result['education']}")
                        st.write(f"**Work Experience:** {result['experience']}")
                        st.write(f"**Current Background:** {result['current_title']}")
                        st.write(f"**Skills:** {result['raw_data'][2][:100]}...")

                    with col2:
                        st.write("**Match Analysis:**")
                        st.write(f"**Match Score:** {score_color} {result['match_score']} points")
                        st.write(f"**Salary Match:** {result['analysis'].get('salary_match', 'Average')}")
                        st.write(f"**Culture Fit:** {result['analysis'].get('culture_fit', 'Medium')}")

                        if 'key_strengths' in result['analysis']:
                            st.write("**Core Strengths:**")
                            for strength in result['analysis']['key_strengths']:
                                st.write(f"✅ {strength}")

                        if 'potential_gaps' in result['analysis']:
                            st.write("**Areas of Concern:**")
                            for gap in result['analysis']['potential_gaps']:
                                st.write(f"⚠️ {gap}")

                    if 'recommendation' in result['analysis']:
                        st.info(f"**Recommendation:** {result['analysis']['recommendation']}")

                    # Action buttons
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        if st.button("📞 Contact Candidate", key=f"contact_{result['seeker_id']}"):
                            st.success(f"Marked for contact: {result['name']}")
                    with col_btn2:
                        if st.button("💼 Schedule Interview", key=f"interview_{result['seeker_id']}"):
                            st.success(f"Interview scheduled: {result['name']}")
        else:
            st.warning("😔 No matching candidates found, please adjust matching conditions")

def ai_interview_dashboard():
    """AI Interview Dashboard"""
    st.title("🤖 AI Mock Interview System")

    # Quick statistics
    jobs = get_jobs_for_interview()
    seeker_profile = get_job_seeker_profile()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Available Positions", len(jobs) if jobs else 0)
    with col2:
        st.metric("Personal Profile", "✅" if seeker_profile else "❌")
    with col3:
        if 'interview' in st.session_state:
            progress = st.session_state.interview['current_question']
            total = st.session_state.interview['total_questions']
            st.metric("Interview Progress", f"{progress}/{total}")
        else:
            st.metric("Interview Status", "Not Started")

    # Page selection
    page_option = st.sidebar.radio(
        "Select Function",
        ["Start Mock Interview", "Interview Preparation Guide", "Instructions"]
    )

    if page_option == "Start Mock Interview":
        ai_interview_page()
    elif page_option == "Interview Preparation Guide":
        show_interview_guidance()
    else:
        show_interview_instructions()

def show_interview_guidance():
    """Display Interview Preparation Guide"""
    st.header("🎯 Interview Preparation Guide")

    st.info("""
    **Interview Preparation Suggestions:**

    ### 📚 Technical Interview Preparation
    1. **Review Core Skills**: Ensure mastery of key technologies required for the position
    2. **Prepare Project Cases**: Prepare 2-3 projects that demonstrate your capabilities
    3. **Practice Coding Problems**: Prepare algorithms and data structures for technical positions

    ### 💼 Behavioral Interview Preparation
    1. **STAR Method**: Situation-Task-Action-Result
    2. **Prepare Success Stories**: Show how you solve problems and create value
    3. **Understand Company Culture**: Research company values and work style

    ### 🎯 Communication Skills
    1. **Clear Expression**: Structure your answers
    2. **Active Listening**: Ensure understanding of question core
    3. **Show Enthusiasm**: Express interest in position and company
    """)

def show_interview_instructions():
    """Display Usage Instructions"""
    st.header("📖 AI Mock Interview Usage Instructions")

    st.info("""
    **AI Mock Interview Function Guide:**

    ### 🚀 Start Interview
    1. **Select Position**: Choose a position from headhunter published positions for mock interview
    2. **Start Interview**: AI will generate relevant questions based on position requirements
    3. **Answer Questions**: Provide detailed answers for each question

    ### 📊 Interview Process
    - **10 Questions**: Includes various types like technical, behavioral, situational
    - **Real-time Evaluation**: AI evaluates quality of each answer
    - **Personalized Questions**: Follow-up questions based on your previous answers

    ### 🎯 Get Feedback
    - **Detailed Scoring**: Specific scoring and feedback for each question
    - **Overall Evaluation**: Complete interview performance summary
    - **Improvement Suggestions**: Targeted career development advice

    **Tip**: Please ensure use in stable network environment for AI to generate questions and evaluate answers normally.
    """)


def tailored_resume_page():
    """AI-powered Tailored Resume Page - Generate job-specific resumes with AI"""
    if not MODULES_AVAILABLE:
        st.error("❌ Tailored Resume modules are not available. Please ensure the modules/ directory is properly installed.")
        return
    
    try:
        # Render CSS styles
        render_styles()
        
        st.markdown('<h1 class="main-header">📝 AI-powered Tailored Resume</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-banner">
            <h3>✨ Create Job-Specific Resumes with AI</h3>
            <p>Our AI analyzes job descriptions and tailors your resume to highlight the most relevant skills and experiences.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check if there's a job already selected for resume generation
        if st.session_state.get('show_resume_generator', False) and st.session_state.get('selected_job'):
            modular_display_resume_generator()
            return
        
        # Check if user has profile data
        if not st.session_state.user_profile.get('name'):
            st.warning("⚠️ **Profile Required**: Please complete your profile first to generate tailored resumes.")
            st.info("👉 Go to **Market Dashboard** or **Job Seeker** page to upload your CV and fill in your profile.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Go to Market Dashboard", use_container_width=True):
                    st.session_state.current_page = "market_dashboard"
                    st.rerun()
            with col2:
                if st.button("🏠 Go to Job Seeker", use_container_width=True):
                    st.session_state.current_page = "main"
                    st.rerun()
            return
        
        # Show user profile summary
        st.success(f"✅ Profile loaded for: **{st.session_state.user_profile.get('name', 'N/A')}**")
        
        # Check if there are matched jobs
        if st.session_state.get('matched_jobs') and len(st.session_state.matched_jobs) > 0:
            st.markdown("### 🎯 Select a Job to Tailor Your Resume")
            st.markdown("Choose from your matched jobs below, or search for new jobs in the Market Dashboard.")
            
            # Display matched jobs for selection
            for i, job in enumerate(st.session_state.matched_jobs[:10]):  # Show top 10
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"""
                        **{job.get('title', 'Unknown Title')}**  
                        🏢 {job.get('company', 'Unknown Company')} • 📍 {job.get('location', 'Unknown')}
                        """)
                    with col2:
                        if st.button("✨ Tailor Resume", key=f"tailor_job_{i}", use_container_width=True):
                            st.session_state.selected_job = job
                            st.session_state.show_resume_generator = True
                            st.rerun()
                    st.markdown("---")
        else:
            st.info("💡 **No matched jobs yet.** Go to Market Dashboard to search for jobs and find matches.")
            if st.button("📊 Go to Market Dashboard", use_container_width=True):
                st.session_state.current_page = "market_dashboard"
                st.rerun()
        
        # How it works section
        st.markdown("### 🔧 How It Works")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **1️⃣ Select a Job**  
            Choose a job from your matches or search for new positions.
            """)
        with col2:
            st.markdown("""
            **2️⃣ AI Tailoring**  
            Our AI analyzes the job description and adapts your resume.
            """)
        with col3:
            st.markdown("""
            **3️⃣ Download & Apply**  
            Download as PDF, DOCX, or TXT and apply with confidence!
            """)
        
    except Exception as e:
        st.error(f"❌ An error occurred: {e}")


def market_dashboard_page():
    """Market Dashboard Page - Modular CareerLens Dashboard (from app_new.py)"""
    if not MODULES_AVAILABLE:
        st.error("❌ Market Dashboard modules are not available. Please ensure the modules/ directory is properly installed.")
        st.info("The Market Dashboard requires the modular UI components from the modules/ directory.")
        return
    
    try:
        # Render CSS styles
        render_styles()
        
        # Check if resume generator should be shown
        if st.session_state.get('show_resume_generator', False):
            modular_display_resume_generator()
            return
        
        # Render modular sidebar with controls
        modular_render_sidebar()
        
        # Render hero banner at the top of main content
        render_hero_banner(
            st.session_state.user_profile if st.session_state.user_profile else {},
            st.session_state.matched_jobs if st.session_state.get('dashboard_ready', False) else None
        )
        
        # Main dashboard area - only show after analysis
        if not st.session_state.get('dashboard_ready', False) or not st.session_state.matched_jobs:
            st.info("👆 Upload your CV in the sidebar and click 'Analyze Profile & Find Matches' to see your market positioning and ranked opportunities.")
            return
        
        # Display Market Positioning Profile (Top Section)
        display_market_positioning_profile(
            st.session_state.matched_jobs,
            st.session_state.user_profile
        )
        
        # Display Refine Results Section
        display_refine_results_section(
            st.session_state.matched_jobs,
            st.session_state.user_profile
        )
        
        # Display Smart Ranked Matches Table (Middle Section)
        display_ranked_matches_table(
            st.session_state.matched_jobs,
            st.session_state.user_profile
        )
        
        # Display Match Breakdown & Application Copilot (Bottom Section)
        display_match_breakdown(
            st.session_state.matched_jobs,
            st.session_state.user_profile
        )
    except Exception as e:
        st.error(f"""
        ❌ **Dashboard Error**
        
        An unexpected error occurred: {e}
        
        Please check:
        1. All required secrets are configured
        2. All dependencies are installed
        3. The application logs for more details
        """)

# Sidebar navigation with new design
# Load and embed the logo
def get_logo_base64():
    """Get logo as base64 for embedding in HTML"""
    try:
        with open("Logo.jpg", "rb") as f:
            return base64.b64encode(f.read()).decode()
    except (FileNotFoundError, IOError, OSError):
        # Logo file not found or unreadable
        return None

logo_base64 = get_logo_base64()
logo_html = ""
if logo_base64:
    logo_html = f'<img src="data:image/jpeg;base64,{logo_base64}" style="width: 100%; max-width: 300px; margin: 0 auto 1rem auto; display: block;">'

st.sidebar.markdown(f"""
<style>
    /* CareerLens Logo and Branding */
    .careerlens-logo {{
        font-family: 'Montserrat', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }}
    .careerlens-logo .brand-span {{
        color: var(--brand-core);
    }}
    .careerlens-logo .lens-span {{
        color: var(--brand-glow);
    }}
    .careerlens-tagline {{
        font-family: 'Montserrat', sans-serif;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 0.7rem;
        text-align: center;
        margin-bottom: 2rem;
    }}
    
    /* Navigation Section Headers */
    .nav-section-header {{
        font-family: 'Montserrat', sans-serif;
        font-weight: 700;
        font-size: 1.1rem;
        color: white !important;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        padding-left: 0.5rem;
        border-left: 3px solid var(--brand-glow);
    }}
    
    /* Navigation Items */
    .nav-item {{
        font-family: 'Inter', sans-serif;
        color: var(--text-secondary) !important;
        font-size: 0.9rem;
        padding-left: 1.5rem;
        margin: 0.3rem 0;
        cursor: pointer;
    }}
    .nav-item:hover {{
        color: var(--brand-glow) !important;
    }}
</style>

{logo_html}
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Job Seeker Section
st.sidebar.markdown('<div class="nav-section-header">👤 Job Seeker</div>', unsafe_allow_html=True)
if st.sidebar.button("🏠 Job Seeker", use_container_width=True, key="main_btn"):
    st.session_state.current_page = "main"
if st.sidebar.button("💼 Job Matching", use_container_width=True, key="job_matching_btn"):
    st.session_state.current_page = "job_recommendations"
if st.sidebar.button("📝 AI Powered Tailored Resume", use_container_width=True, key="tailored_resume_btn"):
    st.session_state.current_page = "tailored_resume"
if st.sidebar.button("🤖 AI Mock Interview", use_container_width=True, key="ai_interview_btn"):
    st.session_state.current_page = "ai_interview"
if st.sidebar.button("📊 Market Dashboard", use_container_width=True, key="market_dashboard_btn"):
    st.session_state.current_page = "market_dashboard"
if st.sidebar.button("🧠 How This App Works", use_container_width=True, key="how_it_works_btn"):
    st.session_state.current_page = "how_it_works"

st.sidebar.markdown("---")

# Recruiter Section
st.sidebar.markdown('<div class="nav-section-header">🎯 Recruiter</div>', unsafe_allow_html=True)
if st.sidebar.button("📋 Job Posting", use_container_width=True, key="job_posting_btn"):
    st.session_state.current_page = "head_hunter"
if st.sidebar.button("🔍 Recruitment Match", use_container_width=True, key="recruitment_match_btn"):
    st.session_state.current_page = "recruitment_match"

st.sidebar.markdown("---")

# Add CareerLens tools
with st.sidebar:
    st.subheader("🔍 CareerLens Tools")
    
    # Display domain filter
    if st.session_state.current_page == "job_recommendations":
        with st.expander("🏭 Industry Filters", expanded=False):
            target_domains = st.multiselect(
                "Target Domains",
                options=["FinTech", "ESG & Sustainability", "Data Analytics", "Digital Transformation", 
                        "Investment Banking", "Consulting", "Technology", "Healthcare", "Education"],
                default=st.session_state.get('target_domains', []),
                key="sidebar_domain_filter"
            )
            st.session_state.target_domains = target_domains
            
            salary_exp = st.slider(
                "Min. Salary (HKD)",
                min_value=0,
                max_value=150000,
                value=st.session_state.get('salary_expectation', 0),
                step=5000,
                key="sidebar_salary_filter"
            )
            st.session_state.salary_expectation = salary_exp
    
    # Display token usage
    display_token_usage()
    
    st.markdown("---")
    st.subheader("🔧 Database Debug")
    
    if st.button("View All Job Seeker Records"):
        try:
            conn = sqlite3.connect('job_seeker.db')
            c = conn.cursor()
            c.execute("SELECT job_seeker_id, timestamp, education_level, primary_role FROM job_seekers ORDER BY id DESC")
            results = c.fetchall()
            conn.close()
            
            if results:
                st.write("📋 All Job Seeker Records:")
                for record in results:
                    st.write(f"- ID: {record[0]}, Time: {record[1]}, Education: {record[2]}, Role: {record[3]}")
            else:
                st.write("No job seeker records yet")
        except Exception as e:
            st.error(f"Query failed: {e}")
    
    # Display current session state
    current_id = st.session_state.get('job_seeker_id')
    if current_id:
        st.info(f"Current Session ID: **{current_id}**")

# Page routing
if st.session_state.current_page == "main":
    main_analyzer_page()
elif st.session_state.current_page == "job_recommendations":
    job_seeker_id = st.session_state.get('job_seeker_id')

    # Check if there is saved job seeker data
    if not job_seeker_id:
        st.warning("⚠️ Please first save your personal information on the Job Seeker page")
        st.info("👉 Switch to 'Job Seeker' page to fill in and save your information")
        
        # Provide quick jump
        if st.button("Go to Job Seeker Page"):
            st.session_state.current_page = "main"
            st.rerun()
    else:
        # Call job recommendations page function
        job_recommendations_page(job_seeker_id)

elif st.session_state.current_page == "head_hunter":
    enhanced_head_hunter_page()
elif st.session_state.current_page == "recruitment_match":
    recruitment_match_dashboard()
elif st.session_state.current_page == "ai_interview":
    ai_interview_dashboard()
elif st.session_state.current_page == "tailored_resume":
    tailored_resume_page()
elif st.session_state.current_page == "market_dashboard":
    market_dashboard_page()
elif st.session_state.current_page == "how_it_works":
    render_how_it_works_page()


# Sidebar information
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 💡 Usage Instructions

**For Job Seekers:**
- **Job Seeker**: Upload your CV and fill in your profile
- **Job Matching**: Find AI-matched positions based on your profile
- **AI Powered Tailored Resume**: Generate job-specific resumes
- **AI Mock Interview**: Practice with AI-powered mock interviews
- **Market Dashboard**: View comprehensive market insights
- **How This App Works**: Learn about our AI technology

**For Recruiters:**
- **Job Posting**: Publish and manage job openings
- **Recruitment Match**: Smart candidate-position matching
""")
                    
# Footer
st.markdown("---")
st.caption("🤖 Powered by GPT-4, Pinecone Vector Search, RapidAPI LinkedIn Jobs, and CareerLens AI")

# Application startup
if __name__ == "__main__":
    # Ensure application runs normally
    pass