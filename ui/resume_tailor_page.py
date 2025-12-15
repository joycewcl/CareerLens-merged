"""
AI-Powered Tailored Resume Page.

This module contains the page for generating job-specific resumes with AI:
- Resume tailoring based on job descriptions
- Download options (PDF, DOCX, TXT)

Updated Data Flow (Step 3a):
    Tailored Resume Generation:
       ↓ Retrieve from job_post_API.db
       ↓ WHERE job_seeker_id = current_user
       ↓ Get specific job details
       ↓ Merge with job_seeker.db profile
       ↓ Generate optimized resume
       
Flow:
    ui/resume_tailor_page.py
      ↓
    core/job_seeker_flow.py
      → get_matched_jobs_for_seeker()
      → get_job_for_resume_tailoring()
      ↓
    core/resume_parser.py
      → generate_tailored_resume()
      ↓
    core/resume_generator.py
      → generate_docx_from_json()
      → generate_pdf_from_json()
      → format_resume_as_text()
"""

import streamlit as st
from typing import Dict, List, Optional


def tailored_resume_page():
    """AI-powered Tailored Resume Page - Generate job-specific resumes with AI
    
    This page follows the updated data flow:
    1. Retrieve matched jobs from job_post_API.db for current job_seeker_id
    2. Allow user to select a job for resume tailoring
    3. Generate optimized resume by merging job details with profile
    """
    # Check if modules are available
    try:
        from ui.components.styles import render_styles
        from ui.components.resume_editor import display_resume_generator as modular_display_resume_generator
        MODULES_AVAILABLE = True
    except ImportError:
        MODULES_AVAILABLE = False
    
    # Import job seeker flow module
    try:
        from core.job_seeker_flow import (
            get_matched_jobs_for_seeker,
            get_job_seeker_profile,
            MATCH_SCORE_THRESHOLD
        )
        from database import get_job_seeker_db
        FLOW_AVAILABLE = True
    except ImportError:
        FLOW_AVAILABLE = False
    
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
        
        # Get current job seeker ID
        job_seeker_id = st.session_state.get('job_seeker_id')
        
        # Hydrate user_profile from DB if missing (critical for resume generator)
        if job_seeker_id and FLOW_AVAILABLE:
            if not st.session_state.get('user_profile') or not st.session_state.user_profile.get('name'):
                try:
                    profile_data = get_job_seeker_profile(job_seeker_id)
                    if profile_data:
                        # Construct a basic profile from DB data to satisfy the resume generator
                        st.session_state.user_profile = {
                            'name': f"Job Seeker {job_seeker_id[-6:]}", 
                            'email': '',
                            'phone': '',
                            'location': profile_data.get('location_preference', ''),
                            'linkedin': '',
                            'summary': f"Professional with {profile_data.get('work_experience', '')} experience in {profile_data.get('industry_preference', '')}.",
                            'experience': profile_data.get('work_experience', ''),
                            'education': f"{profile_data.get('education_level', '')} in {profile_data.get('major', '')}",
                            'skills': profile_data.get('hard_skills', ''),
                            'certifications': profile_data.get('certificates', '')
                        }
                except Exception as e:
                    print(f"Error hydrating profile: {e}")

        # Check if there's a job already selected for resume generation
        if st.session_state.get('show_resume_generator', False) and st.session_state.get('selected_job'):
            modular_display_resume_generator()
            return
        
        # Check if user has job_seeker_id (profile saved)
        if not job_seeker_id:
            st.warning("⚠️ **Profile Required**: Please complete your profile first to generate tailored resumes.")
            st.info("👉 Go to **Job Seeker** page to upload your CV and save your profile.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Go to Job Search", use_container_width=True):
                    st.session_state.current_page = "job_recommendations"
                    st.rerun()
            with col2:
                if st.button("🏠 Go to Job Seeker", use_container_width=True):
                    st.session_state.current_page = "main"
                    st.rerun()
            return
        
        # Show job seeker ID
        st.success(f"✅ Profile loaded: **{job_seeker_id}**")
        
        # ==========================================
        # STEP 3a DATA FLOW: Retrieve matched jobs from job_post_API.db
        # ==========================================
        saved_jobs = []
        if FLOW_AVAILABLE:
            try:
                saved_jobs = get_matched_jobs_for_seeker(job_seeker_id, min_score=0.0, limit=20)
            except Exception as e:
                st.warning(f"Could not retrieve saved jobs: {e}")
        
        # Also check session state for recent matches (may not be saved yet)
        session_jobs = st.session_state.get('matched_jobs', [])
        
        # Display saved jobs from job_post_API.db (priority)
        if saved_jobs and len(saved_jobs) > 0:
            st.markdown("### 💾 Your Saved Matched Jobs")
            st.caption(f"Jobs from your profile with match scores above {MATCH_SCORE_THRESHOLD}%")
            
            _display_job_selection(saved_jobs, source="db")
            
        # Also show session jobs if different from saved
        elif session_jobs and len(session_jobs) > 0:
            st.markdown("### 🎯 Select a Job to Tailor Your Resume")
            st.markdown("Choose from your matched jobs below, or search for new jobs.")
            
            _display_job_selection(session_jobs, source="session")
            
        else:
            st.info("💡 **No matched jobs yet.** Go to Job Search to find and match jobs.")
            st.markdown("""
            ### How to get matched jobs:
            1. Go to **Job Search** page
            2. Search for jobs matching your profile
            3. Jobs with match score ≥60% will be saved automatically
            4. Return here to generate tailored resumes
            """)
            if st.button("🔍 Go to Job Search", use_container_width=True):
                st.session_state.current_page = "job_recommendations"
                st.rerun()
        
        # How it works section
        st.markdown("---")
        st.markdown("### 🔧 How It Works")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **1️⃣ Select a Job**  
            Choose from your saved matches (jobs with ≥60% match are saved automatically).
            """)
        with col2:
            st.markdown("""
            **2️⃣ AI Tailoring**  
            Our AI merges your profile with job requirements to optimize your resume.
            """)
        with col3:
            st.markdown("""
            **3️⃣ Download & Apply**  
            Download as PDF, DOCX, or TXT and apply with confidence!
            """)
        
    except Exception as e:
        st.error(f"❌ An error occurred: {e}")


def _display_job_selection(jobs: List[Dict], source: str = "session"):
    """Display job selection list for resume tailoring.
    
    Args:
        jobs: List of job dictionaries
        source: "db" for jobs from job_post_API.db, "session" for session state jobs
    """
    for i, job in enumerate(jobs[:10]):  # Show top 10
        # Handle different data structures
        if source == "db":
            # Jobs from job_post_API.db have different field names
            title = job.get('job_title', 'Unknown Title')
            company = job.get('company_name', 'Unknown Company')
            location = job.get('location', 'Unknown')
            match_pct = job.get('match_percentage', 0)
            job_id = job.get('job_id', '')
        else:
            # Jobs from session state
            job_data = job.get('job', job)
            title = job_data.get('title', 'Unknown Title')
            company = job_data.get('company', 'Unknown Company')
            location = job_data.get('location', 'Unknown')
            match_pct = job.get('combined_score', job.get('match_percentage', 0))
            job_id = job.get('id', '')
        
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"""
                **{title}**  
                🏢 {company} • 📍 {location}
                """)
            with col2:
                # Display match score
                if match_pct >= 80:
                    st.markdown(f"🟢 **{match_pct:.0f}%**")
                elif match_pct >= 60:
                    st.markdown(f"🟡 **{match_pct:.0f}%**")
                else:
                    st.markdown(f"🟠 **{match_pct:.0f}%**")
            with col3:
                if st.button("✨ Tailor", key=f"tailor_{source}_{i}", use_container_width=True):
                    # Prepare job for resume generator
                    if source == "db":
                        # Convert DB format to expected format
                        selected_job = {
                            'id': job_id,
                            'title': title,
                            'company': company,
                            'location': location,
                            'description': job.get('job_description', ''),
                            'skills': [s.strip() for s in job.get('required_skills', '').split(',') if s.strip()],
                            'job_type': job.get('employment_type', ''),
                            'url': job.get('application_url', ''),
                            # Store original DB record for reference
                            '_db_record': job
                        }
                    else:
                        selected_job = job.get('job', job)
                        selected_job['id'] = job_id
                    
                    st.session_state.selected_job = selected_job
                    st.session_state.show_resume_generator = True
                    st.rerun()
            st.markdown("---")
