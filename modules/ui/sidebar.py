"""Sidebar UI component"""
import streamlit as st
import time
import os
from modules.resume_upload import extract_text_from_resume, extract_profile_from_resume
from modules.semantic_search import generate_and_store_resume_embedding
from .dashboard import display_skill_matching_matrix
from modules.utils.helpers import get_img_as_base64


def _get_sidebar_logo_html():
    """Get logo HTML for sidebar"""
    logo_configs = [
        ("CareerLens_Logo.png", "image/png"),
        ("Logo.jpg", "image/jpeg")
    ]
    
    for logo_path, mime_type in logo_configs:
        if os.path.exists(logo_path):
            try:
                logo_base64 = get_img_as_base64(logo_path)
                return f'<img src="data:{mime_type};base64,{logo_base64}" class="sidebar-logo-image">'
            except Exception as e:
                # Failed to load logo file - continue to next
                continue
    # No logo found - return empty string
    return ''


def render_sidebar():
    """Render CareerLens sidebar with resume upload and search criteria settings.
    
    Note: Search functionality is now handled only in the dashboard (display_refine_results_section).
    The sidebar is simplified to only handle profile upload and filter settings.
    """
    with st.sidebar:
        logo_html = _get_sidebar_logo_html()
        
        st.markdown(f"""
        <style>
            .sidebar-logo {{
                color: white !important;
                margin-bottom: 0.5rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
                font-family: 'Montserrat', sans-serif;
                font-size: 2rem;
                font-weight: 700;
                letter-spacing: -1px;
                text-align: center;
                justify-content: center;
            }}
            .sidebar-logo .brand-span {{
                color: var(--brand-core);
            }}
            .sidebar-logo .lens-span {{
                color: var(--brand-glow);
            }}
            .sidebar-tagline {{
                color: var(--text-secondary-light);
                font-size: 0.7rem;
                margin: 0;
                font-family: 'Montserrat', sans-serif;
                text-transform: uppercase;
                letter-spacing: 2px;
                text-align: center;
            }}
        </style>
        <div style="margin-bottom: 2rem;">
            {logo_html}
            <h2 class="sidebar-logo">
                <span class="brand-span">Career</span><span class="lens-span">Lens</span>
            </h2>
            <p class="sidebar-tagline">AI Career Copilot • Hong Kong</p>
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
                            'hard_skills': profile_data.get('skills', ''),  # Alias for compatibility
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
        st.info("💡 Click **Refine Results** in the dashboard to search for jobs with your criteria.")
        
        # Display skill matching explanation
        display_skill_matching_matrix(st.session_state.user_profile)
