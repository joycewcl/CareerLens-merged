"""
AI-Powered Tailored Resume Page.

This module contains the page for generating job-specific resumes with AI:
- Resume tailoring based on job descriptions
- Download options (PDF, DOCX, TXT)

Flow:
    ui/resume_tailor_page.py
      ↓
    services/azure_openai.py
      → TextGenerator.generate_resume()
      ↓
    modules/resume_generator/formatters.py
      → generate_docx_from_json()
      → generate_pdf_from_json()
      → format_resume_as_text()
"""

import streamlit as st


def tailored_resume_page():
    """AI-powered Tailored Resume Page - Generate job-specific resumes with AI"""
    # Check if modules are available
    try:
        from modules.ui.styles import render_styles
        from modules.ui import display_resume_generator as modular_display_resume_generator
        MODULES_AVAILABLE = True
    except ImportError:
        MODULES_AVAILABLE = False
    
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
        if not st.session_state.get('user_profile', {}).get('name'):
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
