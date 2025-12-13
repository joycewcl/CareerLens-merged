"""Sidebar UI component"""
import streamlit as st
import time
import gc
from modules.resume_upload import extract_text_from_resume, extract_profile_from_resume
from modules.semantic_search import (
    SemanticJobSearch,
    fetch_jobs_with_cache,
    generate_and_store_resume_embedding
)
from modules.analysis import filter_jobs_by_domains, filter_jobs_by_salary
from modules.utils import get_embedding_generator, get_job_scraper, _websocket_keepalive, _ensure_websocket_alive
from modules.utils.config import _determine_index_limit
from .dashboard import display_skill_matching_matrix


def render_sidebar():
    """Render CareerLens sidebar with resume upload, market filters, and analyze button"""
    with st.sidebar:
        st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Montserrat:wght@400;700&display=swap');
        </style>
        <div style="margin-bottom: 2rem;">
            <h2 style="color: white !important; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem; font-family: 'Montserrat', sans-serif; font-size: 2rem; font-weight: 700; letter-spacing: -1px; text-align: center; justify-content: center;">
                <span style="color: #0084C2;">Career</span><span style="color: #00D2FF;">Lens</span>
            </h2>
            <p style="color: #94a3b8 !important; font-size: 0.7rem; margin: 0; font-family: 'Montserrat', sans-serif; text-transform: uppercase; letter-spacing: 2px; text-align: center;">AI Career Copilot • Hong Kong</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 1. Upload your CV to begin")
        uploaded_file = st.file_uploader(
            "Upload your resume",
            type=['pdf', 'docx'],
            help="We parse your skills and experience to benchmark you against the market.",
            key="careerlens_resume_upload",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            file_key = f"{uploaded_file.name}_{uploaded_file.size}"
            current_cached_key = st.session_state.get('_last_uploaded_file_key')
            
            if current_cached_key != file_key:
                progress_bar = st.progress(0, text="📖 Reading resume...")
                resume_text = extract_text_from_resume(uploaded_file)
                
                if resume_text:
                    progress_bar.progress(30, text="✅ Resume read successfully")
                    st.session_state.resume_text = resume_text
                    st.session_state._last_uploaded_file_key = file_key
                    
                    progress_bar.progress(40, text="🤖 Extracting profile with AI...")
                    profile_data = extract_profile_from_resume(resume_text)
                    
                    if profile_data:
                        progress_bar.progress(80, text="📊 Finalizing profile...")
                        st.session_state.user_profile = {
                            'name': profile_data.get('name', ''),
                            'email': profile_data.get('email', ''),
                            'phone': profile_data.get('phone', ''),
                            'location': profile_data.get('location', ''),
                            'linkedin': profile_data.get('linkedin', ''),
                            'portfolio': profile_data.get('portfolio', ''),
                            'summary': profile_data.get('summary', ''),
                            'experience': profile_data.get('experience', ''),
                            'education': profile_data.get('education', ''),
                            'skills': profile_data.get('skills', ''),
                            'certifications': profile_data.get('certifications', '')
                        }
                        
                        progress_bar.progress(90, text="🔗 Creating search embedding...")
                        generate_and_store_resume_embedding(resume_text, st.session_state.user_profile)
                        
                        progress_bar.progress(100, text="✅ Profile ready!")
                        time.sleep(0.3)
                        progress_bar.empty()
                        st.success("✅ Profile extracted!")
                    else:
                        progress_bar.empty()
                        st.warning("⚠️ Could not extract profile. Please try again.")
                else:
                    progress_bar.empty()
                    st.error("❌ Could not read the resume file.")
            else:
                if st.session_state.user_profile.get('name'):
                    st.success(f"✅ Using profile for: {st.session_state.user_profile.get('name', 'Unknown')}")
        
        st.markdown("---")
        st.markdown("### 2. Set Search Criteria")
        
        target_domains = st.multiselect(
            "Target Domains",
            options=["FinTech", "ESG & Sustainability", "Data Analytics", "Digital Transformation", 
                    "Investment Banking", "Consulting", "Technology", "Healthcare", "Education"],
            default=st.session_state.get('target_domains', []),
            help="Select industries/domains to search for jobs",
            key="sidebar_target_domains"
        )
        st.session_state.target_domains = target_domains
        
        salary_expectation = st.slider(
            "Min. Monthly Salary (HKD)",
            min_value=0,
            max_value=150000,
            value=st.session_state.get('salary_expectation', 0),
            step=5000,
            help="Set to 0 to disable salary filtering",
            key="sidebar_salary"
        )
        st.session_state.salary_expectation = salary_expectation
        
        st.markdown("---")
        analyze_button = st.button(
            "Analyze Profile & Find Matches",
            type="primary",
            use_container_width=True,
            key="careerlens_analyze"
        )
        
        if analyze_button:
            if not st.session_state.resume_text and not st.session_state.user_profile.get('summary'):
                st.error("⚠️ Please upload your CV first!")
            else:
                target_domains = st.session_state.get('target_domains', [])
                salary_expectation = st.session_state.get('salary_expectation', 0)
                
                search_query = " ".join(target_domains) if target_domains else "Hong Kong jobs"
                scraper = get_job_scraper()
                
                if scraper is None:
                    st.error("⚠️ Job scraper not configured. Please check your RAPIDAPI_KEY in Streamlit secrets.")
                    return
                
                progress_bar = st.progress(0, text="🔍 Starting job search...")
                
                progress_bar.progress(10, text="📡 Fetching jobs from Indeed...")
                _websocket_keepalive("Connecting to job API...")
                
                jobs = fetch_jobs_with_cache(
                    scraper,
                    search_query,
                    location="Hong Kong",
                    max_rows=25,
                    job_type="fulltime",
                    country="hk",
                    force_refresh=False
                )
                
                _websocket_keepalive("Processing job results...")
                
                if not jobs:
                    progress_bar.empty()
                    st.error("❌ No jobs found from Indeed. Please check your API configuration or try different search criteria.")
                    return
                
                total_fetched = len(jobs)
                progress_bar.progress(30, text=f"✅ Found {total_fetched} jobs, applying filters...")
                _websocket_keepalive()
                
                if target_domains:
                    jobs = filter_jobs_by_domains(jobs, target_domains)
                
                if salary_expectation > 0:
                    jobs = filter_jobs_by_salary(jobs, salary_expectation)
                
                if not jobs:
                    progress_bar.empty()
                    st.warning(f"⚠️ No jobs match your filters. Found {total_fetched} jobs but none passed your criteria. Try reducing salary or selecting different domains.")
                    return
                
                progress_bar.progress(40, text=f"📊 Analyzing {len(jobs)} matching jobs...")
                _websocket_keepalive("Initializing analysis engine...")
                
                embedding_gen = get_embedding_generator()
                if embedding_gen is None:
                    progress_bar.empty()
                    st.error("⚠️ Azure OpenAI is not configured.")
                    return
                
                desired_matches = min(15, len(jobs))
                jobs_to_index_limit = _determine_index_limit(len(jobs), desired_matches)
                top_match_count = min(desired_matches, jobs_to_index_limit)
                search_engine = SemanticJobSearch(embedding_gen)
                
                progress_bar.progress(50, text=f"🔗 Creating job embeddings ({jobs_to_index_limit} jobs)...")
                _websocket_keepalive("Creating job embeddings...")
                search_engine.index_jobs(jobs, max_jobs_to_index=jobs_to_index_limit)
                
                _ensure_websocket_alive()
                
                resume_embedding = st.session_state.get('resume_embedding')
                if not resume_embedding and st.session_state.resume_text:
                    progress_bar.progress(70, text="🔗 Creating resume embedding...")
                    _websocket_keepalive("Creating resume embedding...")
                    resume_embedding = generate_and_store_resume_embedding(
                        st.session_state.resume_text,
                        st.session_state.user_profile if st.session_state.user_profile else None
                    )
                
                resume_query = None
                if not resume_embedding:
                    if st.session_state.resume_text:
                        resume_query = st.session_state.resume_text
                        if st.session_state.user_profile.get('summary'):
                            profile_data = f"{st.session_state.user_profile.get('summary', '')} {st.session_state.user_profile.get('experience', '')} {st.session_state.user_profile.get('skills', '')}"
                            resume_query = f"{resume_query} {profile_data}"
                    else:
                        resume_query = f"{st.session_state.user_profile.get('summary', '')} {st.session_state.user_profile.get('experience', '')} {st.session_state.user_profile.get('skills', '')} {st.session_state.user_profile.get('education', '')}"
                
                progress_bar.progress(80, text="🎯 Finding best matches...")
                results = search_engine.search(query=resume_query, top_k=top_match_count, resume_embedding=resume_embedding)
                
                if results:
                    progress_bar.progress(90, text="📈 Calculating skill matches...")
                    _websocket_keepalive("Calculating skill matches...")
                    user_skills = st.session_state.user_profile.get('skills', '')
                    total_results = len(results)
                    for i, result in enumerate(results):
                        # Send keepalive every few jobs to prevent timeout
                        if i % 3 == 0:
                            _ensure_websocket_alive()
                        
                        job_skills = result['job'].get('skills', [])
                        skill_score, missing_skills = search_engine.calculate_skill_match(user_skills, job_skills)
                        result['skill_match_score'] = skill_score
                        result['missing_skills'] = missing_skills
                        
                        semantic_score = result.get('similarity_score', 0.0)
                        combined_score = (semantic_score * 0.6) + (skill_score * 0.4)
                        result['combined_match_score'] = combined_score
                        
                        # Update progress during skill matching
                        match_progress = 90 + int((i + 1) / total_results * 9)
                        progress_bar.progress(match_progress, text=f"📈 Analyzing job {i + 1}/{total_results}...")
                    
                    results.sort(key=lambda x: x.get('combined_match_score', 0.0), reverse=True)
                    
                    progress_bar.progress(100, text="✅ Analysis complete!")
                    _websocket_keepalive("Analysis complete!")
                    time.sleep(0.3)
                    progress_bar.empty()
                    
                    st.session_state.matched_jobs = results
                    st.session_state.dashboard_ready = True
                    
                    gc.collect()
                    
                    st.rerun()
                else:
                    progress_bar.empty()
                    st.error("❌ No matching jobs found. Please try different filters.")
        
        display_skill_matching_matrix(st.session_state.user_profile)
