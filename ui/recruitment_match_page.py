"""
Recruitment Match Dashboard - Smart Talent Matching.

This module contains pages for matching candidates with job positions:
- Smart Talent Matching using AI
- Match Statistics
- Usage Instructions
"""

import streamlit as st
from typing import List, Dict


def recruitment_match_dashboard():
    """Recruitment Match Dashboard"""
    from database.queries import get_all_jobs_for_matching, get_all_job_seekers
    
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
    from database.queries import get_all_jobs_for_matching, get_all_job_seekers
    from core.job_matcher import analyze_match_simple
    
    # Import WebSocket utilities with fallback
    try:
        from utils.helpers import ProgressTracker, _websocket_keepalive
    except ImportError:
        def _websocket_keepalive(message=None, force=False):
            pass
        
        class ProgressTracker:
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


def show_match_statistics():
    """Show match statistics in Streamlit UI."""
    from database.queries import get_all_jobs_for_matching_tuples, get_all_job_seekers_formatted
    
    st.header("📊 Match Statistics")

    jobs = get_all_jobs_for_matching_tuples()
    seekers = get_all_job_seekers_formatted()

    if not jobs or not seekers:
        st.info("No statistics data available")
        return

    # Industry distribution
    st.subheader("🏭 Industry Distribution")
    industry_counts = {}
    for job in jobs:
        industry = job[6] if job[6] else "Not Specified"
        industry_counts[industry] = industry_counts.get(industry, 0) + 1

    for industry, count in industry_counts.items():
        percentage = (count / len(jobs)) * 100
        st.write(f"• **{industry}:** {count} Positions ({percentage:.1f}%)")

    # Experience Level Distribution
    st.subheader("🎯 Experience Level Distribution")
    experience_counts = {}
    for job in jobs:
        experience = job[11] if len(job) > 11 and job[11] else "Not Specified"
        experience_counts[experience] = experience_counts.get(experience, 0) + 1

    for exp, count in experience_counts.items():
        st.write(f"• **{exp}:** {count} Positions")


def show_instructions():
    """Display usage instructions in Streamlit UI."""
    st.header("📖 Instructions")

    st.info("""
    **Recruitment Match Instructions:**

    1. **Select Position**: Choose a position from the positions published by the headhunter module
    2. **Set Conditions**: Adjust the minimum match score and display count
    3. **Start Matching**: The system will automatically analyze the match between all job seekers and the position
    4. **View Results**: View detailed match analysis report
    5. **Take Action**: Contact candidates, schedule interviews

    **Matching Algorithm Based on:**
    • Skill Match (Hard Skills)
    • Experience Fit (Work Experience Years)
    • Industry Relevance (Industry Preferences)
    • Location Match (Work Location Preferences)
    • Comprehensive Assessment Analysis

    **Data Sources:**
    • Position Information: Positions published by Head Hunter module
    • Job Seeker Information: Information filled in Job Seeker page
    """)
