"""
Job Search and Recommendations Page.

This module contains the job recommendations page where job seekers can:
- Search for jobs using semantic search
- View AI-ranked job matches
- See skill matching analysis
- Generate tailored resumes for specific jobs

Flow:
    ui/job_search_page.py
      ↓
    services/indeed_api.py (or modules/utils/api_clients.py)
      → IndeedJobScraper.search_jobs()
      ↓
    modules/semantic_search/job_search.py
      → SemanticJobSearch.search()
      ↓
    core/job_matcher.py
      → JobMatcher.analyze_simple_match()
"""

import streamlit as st
import time
from typing import List, Dict, Optional


def job_recommendations_page(job_seeker_id: Optional[str] = None):
    """Job Recommendations Page - Using Real API Data"""
    # Import dependencies inside function to avoid circular imports
    from database import JobSeekerDB
    
    # Import WebSocket utilities with fallback
    try:
        from utils.helpers import ProgressTracker, _websocket_keepalive
        from core.semantic_search import SemanticJobSearch, fetch_jobs_with_cache
        from utils import get_embedding_generator, get_job_scraper
        from utils.config import _determine_index_limit
        from ui.components.dashboard import (
            display_market_positioning_profile,
            calculate_match_scores
        )
        MODULES_AVAILABLE = True
    except ImportError:
        MODULES_AVAILABLE = False
        
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

    # Import visualization functions
    from ui.visualizations import create_enhanced_visualizations, create_job_comparison_radar
    
    # Import resume generator UI
    try:
        from core.resume_parser import generate_tailored_resume
        from services.azure_openai import generate_docx_from_json, generate_pdf_from_json, format_resume_as_text
        from core.resume_parser import verify_profile_data_pass2
        RESUME_AVAILABLE = True
    except ImportError:
        RESUME_AVAILABLE = False

    # Cache database instance
    @st.cache_resource
    def get_job_seeker_db():
        return JobSeekerDB()
    
    db = get_job_seeker_db()
    
    st.title("💼 Personalized Job Recommendations")

    # Get job seeker data - add error handling
    job_seeker_data = dict()
    try:
        if job_seeker_id:
            job_seeker_data = db.get_profile(job_seeker_id)
        else:
            # If no ID provided, try to get latest record
            job_seeker_data = db.get_latest_profile()
            
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
    # 🔎 STEP 0: Check for Cached Matches (Optimization)
    # -------------------------------------------------------
    use_cached = False
    cached_matches = []
    current_job_seeker_id = st.session_state.get('job_seeker_id')
    
    if current_job_seeker_id:
        try:
            from database.job_post_api_db import MatchedJobsDB
            matched_db = MatchedJobsDB()
            cache_info = matched_db.get_recent_match_info(current_job_seeker_id, max_age_hours=24)
            
            if cache_info and cache_info.get('count', 0) > 0:
                st.info(f"💾 **Cached Results Available:** {cache_info['count']} matched jobs from {cache_info['newest_match']}")
                
                col_cache1, col_cache2 = st.columns(2)
                with col_cache1:
                    if st.button("📂 Use Cached Results", use_container_width=True):
                        use_cached = True
                        cached_matches = matched_db.get_matched_jobs(
                            current_job_seeker_id, 
                            min_match=0, 
                            purpose='general',
                            limit=50
                        )
                with col_cache2:
                    st.caption("Or search for new jobs below ↓")
        except Exception as e:
            # Silently continue if cache check fails
            pass
    
    # -------------------------------------------------------
    # 🔎 STEP 2: Search Jobs using Unified SemanticJobSearch
    # -------------------------------------------------------
    # Search button to trigger job search
    search_button = st.button("🔍 Search New Jobs", type="primary", use_container_width=True)
    
    matched_jobs = st.session_state.get('matched_jobs', [])
    
    # Handle cached results (Step 0)
    if use_cached and cached_matches:
        st.success(f"✅ Loaded {len(cached_matches)} cached matched jobs!")
        # Convert DB format to display format
        matched_jobs = []
        for job in cached_matches:
            matched_jobs.append({
                'job': {
                    'title': job.get('job_title', ''),
                    'company': job.get('company_name', ''),
                    'location': job.get('location', ''),
                    'description': job.get('job_description', ''),
                    'skills': [s.strip() for s in job.get('required_skills', '').split(',') if s.strip()],
                    'url': job.get('application_url', ''),
                    'posted_date': job.get('posted_date', ''),
                    'job_type': job.get('employment_type', ''),
                    'id': job.get('job_id', '')
                },
                'combined_score': job.get('match_percentage', 0),
                'semantic_score': (job.get('cosine_similarity_score', 0) or 0) * 100,
                'skill_match_percentage': (job.get('skill_match_score', 0) or 0) * 100,
                'matched_skills': [s.strip() for s in job.get('matched_skills', '').split(',') if s.strip()],
                'missing_skills': [s.strip() for s in job.get('missing_skills', '').split(',') if s.strip()],
            })
        st.session_state.matched_jobs = matched_jobs
    elif search_button:
        if not MODULES_AVAILABLE:
            st.error("⚠️ Search modules not available. Please check your installation.")
            matched_jobs = []
        else:
            # Use ProgressTracker to maintain WebSocket connection during long operations
            with ProgressTracker(f"Searching {num_jobs_to_search} jobs", total_steps=6) as tracker:
                try:
                    # ----------------------------------------------------
                    # 1) Prepare search query
                    # ----------------------------------------------------
                    tracker.update(1, "Preparing search query...")
                    _websocket_keepalive("Preparing query")
                    
                    hard_skills = job_seeker_data.get("hard_skills", "")
                    
                    # Build search keywords from user input
                    search_keywords = search_query if search_query.strip() else job_seeker_data.get("primary_role", "")
                    if not search_keywords:
                        search_keywords = job_seeker_data.get("simple_search_terms", "Hong Kong jobs")
                    
                    location_preference = location if location else job_seeker_data.get("location_preference", "Hong Kong")
                    
                    st.info(
                        f"📡 Searching jobs:\n\n"
                        f"**Keywords:** {search_keywords}\n"
                        f"**Location:** {location_preference}"
                    )
        
                    # ----------------------------------------------------
                    # 2) Fetch jobs using unified job scraper
                    # ----------------------------------------------------
                    tracker.update(2, "Fetching jobs from job boards...")
                    _websocket_keepalive("Connecting to job API")
                    
                    scraper = get_job_scraper()
                    if scraper is None:
                        st.error("⚠️ Job scraper not configured. Please check your RAPIDAPI_KEY.")
                        matched_jobs = []
                    else:
                        jobs = fetch_jobs_with_cache(
                            scraper,
                            search_keywords,
                            location=location_preference,
                            max_rows=num_jobs_to_search,
                            job_type="fulltime" if "FULLTIME" in employment_types else None,
                            country=country,
                            force_refresh=False
                        )
                        
                        if not jobs:
                            st.warning(
                                "⚠️ **No jobs found.** This could be due to:\n\n"
                                "• **Rate limit**: API free tier has limited requests. Wait a few minutes.\n"
                                "• **Search terms**: Try broader keywords.\n"
                                "• **Location**: Try a different location.\n"
                            )
                            matched_jobs = []
                        else:
                            # ----------------------------------------------------
                            # 3) Initialize SemanticJobSearch and index jobs
                            # ----------------------------------------------------
                            tracker.update(3, f"Indexing {len(jobs)} jobs...")
                            _websocket_keepalive("Creating job embeddings")
                            
                            embedding_gen = get_embedding_generator()
                            if embedding_gen is None:
                                st.error("⚠️ Embedding generator not configured.")
                                matched_jobs = []
                            else:
                                desired_matches = min(num_jobs_to_show, len(jobs))
                                jobs_to_index_limit = _determine_index_limit(len(jobs), desired_matches)
                                
                                search_engine = SemanticJobSearch(embedding_gen)
                                search_engine.index_jobs(jobs, max_jobs_to_index=jobs_to_index_limit)
                                
                                # ----------------------------------------------------
                                # 4) Create resume query for semantic search
                                # ----------------------------------------------------
                                tracker.update(4, "Creating search embedding...")
                                _websocket_keepalive("Creating resume embedding")
                                
                                # Build resume query text from job seeker data
                                resume_query = f"""
                                {job_seeker_data.get('primary_role', '')}
                                {job_seeker_data.get('simple_search_terms', '')}
                                {hard_skills}
                                {job_seeker_data.get('soft_skills', '')}
                                {job_seeker_data.get('work_experience', '')}
                                {job_seeker_data.get('project_experience', '')}
                                """
                                
                                # ----------------------------------------------------
                                # 5) Search and rank jobs
                                # ----------------------------------------------------
                                tracker.update(5, "Finding best matches...")
                                _websocket_keepalive("Searching jobs")
                                
                                results = search_engine.search(query=resume_query, top_k=num_jobs_to_show)
                                
                                if results:
                                    # Calculate match scores using unified formula (0-100 scale)
                                    matched_jobs = calculate_match_scores(results, hard_skills)
                                    
                                    # Sort by combined score
                                    matched_jobs.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
                                    
                                    # Save to session state
                                    st.session_state.matched_jobs = matched_jobs
                                    
                                    # ==========================================
                                    # STEP 2 DATA FLOW: Store matched jobs in job_post_API.db
                                    # Only jobs above 60% match threshold are saved
                                    # ==========================================
                                    current_job_seeker_id = st.session_state.get('job_seeker_id')
                                    if current_job_seeker_id:
                                        try:
                                            from core.job_seeker_flow import MATCH_SCORE_THRESHOLD
                                            from database import save_matched_jobs_batch
                                            
                                            # Filter and prepare jobs for storage
                                            jobs_to_store = []
                                            for job in matched_jobs:
                                                combined_score = job.get('combined_score', job.get('combined_match_score', 0))
                                                
                                                # Save all matched jobs regardless of score
                                                # if combined_score >= MATCH_SCORE_THRESHOLD:
                                                job_record = _prepare_job_for_storage(current_job_seeker_id, job)
                                                jobs_to_store.append(job_record)
                                            
                                            # Store in job_post_API.db
                                            if jobs_to_store:
                                                saved_count = save_matched_jobs_batch(jobs_to_store)
                                                st.caption(f"💾 Saved {saved_count} matched jobs to your profile")
                                        except Exception as e:
                                            # Don't block the UI if storage fails
                                            print(f"Warning: Could not store matched jobs: {e}")
                                    
                                    # Save to search history
                                    _save_search_to_history(search_keywords, location_preference, matched_jobs)
                                    
                                    tracker.update(6, "Search complete!")
                                    _websocket_keepalive("Complete")
                                else:
                                    matched_jobs = []
                                    st.warning("⚠️ No matching jobs found. Try different search criteria.")
                    
                except Exception as e:
                    st.error(
                        f"❌ **Search failed:** {str(e)}\n\n"
                        "**Troubleshooting:**\n"
                        "• Check your internet connection\n"
                        "• Verify API keys are configured correctly\n"
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
            _display_job_matches(matched_jobs, num_jobs_to_show, job_seeker_data)

        else:
            st.warning("⚠️ No matched jobs found. Please try adjusting your search criteria.")
            
    if matched_jobs and len(matched_jobs) > 0:
        # Display CareerLens Market Positioning Dashboard
        if MODULES_AVAILABLE:
            display_market_positioning_profile(matched_jobs, job_seeker_data)
        
        # Create enhanced visualizations
        create_enhanced_visualizations(matched_jobs, job_seeker_data)
        
        # Create radar chart comparison for top jobs
        # create_job_comparison_radar(matched_jobs)
        
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


def _prepare_job_for_storage(job_seeker_id: str, job: Dict) -> Dict:
    """
    Prepare a matched job record for storage in job_post_API.db.
    
    This function transforms the matched job data into the format
    expected by the MatchedJobsDB.save_matched_job() method.
    
    Args:
        job_seeker_id: The job seeker's ID
        job: Matched job dictionary with scores
        
    Returns:
        Dictionary ready for database storage
    """
    import uuid
    
    # Handle both nested 'job' structure and flat structure
    job_data = job.get('job', job)
    
    # Generate unique job_id
    job_url = job_data.get('url', '')
    job_title = job_data.get('title', '')
    job_company = job_data.get('company', '')
    job_id = f"JOB_{uuid.uuid4().hex[:12].upper()}"
    
    # Extract scores from result
    cosine_score = job.get('semantic_score', job.get('similarity_score', 0)) / 100.0
    match_percentage = int(job.get('combined_score', job.get('combined_match_score', 0)))
    skill_match_score = job.get('skill_match_percentage', job.get('skill_match_score', 0)) / 100.0
    
    # Get matched and missing skills
    matched_skills = job.get('matched_skills', [])
    missing_skills = job.get('missing_skills', job_data.get('skills', []))
    
    # Handle skills lists
    required_skills = job_data.get('skills', [])
    if isinstance(required_skills, list):
        required_skills_str = ', '.join(required_skills[:20])
    else:
        required_skills_str = str(required_skills)
    
    if isinstance(matched_skills, list):
        matched_skills_str = ', '.join(matched_skills[:20])
    else:
        matched_skills_str = str(matched_skills)
    
    if isinstance(missing_skills, list):
        missing_skills_str = ', '.join(missing_skills[:10])
    else:
        missing_skills_str = str(missing_skills)
    
    return {
        'job_seeker_id': job_seeker_id,
        'job_id': job_id,
        'job_title': job_title,
        'company_name': job_company,
        'location': job_data.get('location', ''),
        'job_description': job_data.get('description', '')[:5000],  # Truncate for storage
        'required_skills': required_skills_str,
        'preferred_skills': '',
        'experience_required': job_data.get('experience_level', ''),
        'salary_min': None,
        'salary_max': None,
        'employment_type': job_data.get('job_type', 'Full-time'),
        'industry': job_data.get('industry', ''),
        'posted_date': job_data.get('posted_date', ''),
        'application_url': job_url,
        'cosine_similarity_score': cosine_score,
        'match_percentage': match_percentage,
        'skill_match_score': skill_match_score,
        'experience_match_score': None,
        'matched_skills': matched_skills_str,
        'missing_skills': missing_skills_str
    }


def _save_search_to_history(search_query: str, location: str, results: List[Dict]):
    """Save search results to history for preservation"""
    if not results:
        return
    
    MAX_SEARCH_HISTORY = 5  # Keep last 5 searches
    
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    search_entry = {
        'timestamp': time.time(),
        'query': search_query,
        'location': location,
        'result_count': len(results),
        'results': results[:10]  # Keep only top 10 for memory efficiency
    }
    
    # Add to beginning of list (most recent first)
    st.session_state.search_history.insert(0, search_entry)
    
    # Limit history size
    if len(st.session_state.search_history) > MAX_SEARCH_HISTORY:
        st.session_state.search_history = st.session_state.search_history[:MAX_SEARCH_HISTORY]


def _display_job_matches(matched_jobs: List[Dict], num_jobs_to_show: int, job_seeker_data: Dict):
    """Display job match results with expandable details"""
    # Import resume generator if available
    try:
        from core.resume_parser import generate_tailored_resume
        from services.azure_openai import generate_docx_from_json, generate_pdf_from_json, format_resume_as_text
        from core.resume_parser import verify_profile_data_pass2
        from utils.helpers import ProgressTracker, _websocket_keepalive
        from ui.visualizations import create_enhanced_visualizations, create_job_comparison_radar
        RESUME_AVAILABLE = True
    except ImportError:
        RESUME_AVAILABLE = False
        
        def _websocket_keepalive(message=None, force=False):
            pass
        
        class ProgressTracker:
            def __init__(self, *args, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def update(self, *args, **kwargs):
                pass

    for i, result in enumerate(matched_jobs[:num_jobs_to_show], start=1):
        # Handle both nested 'job' structure (from SemanticJobSearch) and flat structure
        job = result.get('job', result)
        
        combined = result.get("combined_score", result.get("combined_match_score", 0))

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

        # Check if this job is selected for resume tailoring to keep expanded
        is_selected_for_resume = False
        selected_job = st.session_state.get('selected_job_for_resume')
        if st.session_state.get('show_resume_generator') and selected_job:
            # Check by ID if available, otherwise by title and company
            if job.get('id') is not None and selected_job.get('id') is not None:
                is_selected_for_resume = (str(job.get('id')) == str(selected_job.get('id')))
            else:
                # Fallback check
                is_selected_for_resume = (
                    job.get('title') == selected_job.get('title') and 
                    job.get('company') == selected_job.get('company')
                )

        with st.expander(expander_title, expanded=(i <= 2) or is_selected_for_resume):

            # Scores - use result for scores, job for job data
            semantic_score = result.get('semantic_score', result.get('similarity_score', 0))
            skill_match = result.get('skill_match_percentage', result.get('skill_match_score', 0))
            matched_count = result.get('matched_skills_count', 0)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🎯 Combined Score", f"{combined:.1f}%")
            with col2:
                st.metric("🧠 Semantic Match", f"{semantic_score:.1f}%")
            with col3:
                st.metric("✅ Skill Match", f"{skill_match:.1f}%")
            with col4:
                st.metric("🔢 Skills Matched", matched_count)

            # Job details
            st.markdown("##### 📋 Job Details")
            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:
                st.write(f"**📍 Location:** {job.get('location', 'Unknown')}")
                st.write(f"**🏢 Company:** {job.get('company', 'Unknown')}")

            with detail_col2:
                st.write(f"**📅 Posted:** {job.get('posted_date', 'Unknown')}")
                st.write(f"**💼 Role:** {job.get('title', 'Unknown')}")

            # Matched skills (candidate has) - from result, not job
            matched_skills = result.get("matched_skills", job.get("matched_skills", []))

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
                    if RESUME_AVAILABLE:
                        _display_resume_generator_ui(
                            job, 
                            job_seeker_data,
                            resume_text=st.session_state.get('resume_text')
                        )
                    else:
                        st.warning("Resume generator not available")
                    if st.button("❌ Close Resume Generator", key=f"close_resume_{job.get('id', i)}"):
                        st.session_state.show_resume_generator = False
                        st.session_state.selected_job_for_resume = None
                        st.rerun()

            # Create radar chart for this job, added 15/12/2025 by Michael
            create_job_comparison_radar([result])


def _display_resume_generator_ui(job: Dict, user_profile: Dict, resume_text: str = ""):
    """Display Resume Tailoring UI
    
    Includes Lazy Pass 2: Profile verification runs here before resume generation,
    ensuring accurate dates, company names, and job titles in the tailored resume.
    """
    try:
        from core.resume_parser import generate_tailored_resume
        from services.azure_openai import generate_docx_from_json, generate_pdf_from_json, format_resume_as_text
        from core.resume_parser import verify_profile_data_pass2
        from utils.helpers import ProgressTracker, _websocket_keepalive
    except ImportError:
        st.error("Resume generation modules not available")
        return
    
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
