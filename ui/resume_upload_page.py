"""
Resume Upload Page - CV Upload and Profile Extraction.

This module provides a dedicated page for resume upload and profile extraction
following the modular flow:

Flow:
    ui/resume_upload_page.py
      ↓
    modules/resume_upload/file_extraction.py
      → extract_text_from_resume()
      ↓
    modules/resume_upload/profile_extraction.py
      → extract_profile_from_resume()
      ↓
    core/resume_parser.py
      → ResumeParser.extract_structured_profile()
"""

import streamlit as st
from typing import Dict, Any, Optional


def resume_upload_page():
    """Resume Upload Page - Upload CV and extract profile information.
    
    This page provides a streamlined interface for:
    - Uploading resume files (PDF, DOCX, TXT)
    - Extracting text from uploaded files
    - Using AI to extract structured profile information
    - Storing profile data in session state for use by other pages
    """
    # Import dependencies inside function to avoid circular imports
    from core.resume_parser import ResumeParser, extract_text_from_resume, extract_profile_from_resume
    
    # Import WebSocket utilities with fallback
    try:
        from utils.helpers import ProgressTracker, _websocket_keepalive
    except ImportError:
        # Fallback implementations
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

    # Initialize ResumeParser for structured extraction
    resume_parser = ResumeParser()

    # Page header
    st.markdown("""
    <style>
        .upload-title {
            font-family: 'Montserrat', sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 1rem;
        }
        .upload-subtitle {
            text-align: center;
            color: var(--text-secondary-light);
            font-size: 1rem;
            margin-bottom: 2rem;
        }
    </style>
    <h1 class="upload-title">📄 Resume Upload</h1>
    <p class="upload-subtitle">Upload your CV to extract profile information using AI</p>
    """, unsafe_allow_html=True)

    # File upload section
    st.header("📁 Upload Your Resume")
    
    uploaded_file = st.file_uploader(
        "Choose your resume file",
        type=['pdf', 'docx', 'txt'],
        key="resume_uploader",
        help="Supported formats: PDF, DOCX, TXT"
    )

    if uploaded_file:
        st.success(f"✅ Uploaded: **{uploaded_file.name}**")
        
        # Process button
        if st.button("🔍 Extract Profile Information", type="primary", use_container_width=True):
            _process_resume(
                uploaded_file, 
                extract_text_from_resume,
                extract_profile_from_resume,
                resume_parser,
                ProgressTracker,
                _websocket_keepalive
            )
    else:
        # Welcome/instructions when no file is uploaded
        _display_upload_instructions()

    # Display extracted profile if available
    if st.session_state.get('extracted_profile'):
        _display_extracted_profile()


def _process_resume(
    uploaded_file,
    extract_text_func,
    extract_profile_func,
    resume_parser: 'ResumeParser',
    ProgressTracker,
    _websocket_keepalive
):
    """Process the uploaded resume through the extraction pipeline.
    
    Flow:
        1. extract_text_from_resume() - Extract raw text from file
        2. extract_profile_from_resume() - AI-based profile extraction
        3. ResumeParser.extract_structured_profile() - Additional structured extraction (optional)
    
    Args:
        uploaded_file: Streamlit uploaded file object
        extract_text_func: Function to extract text from file
        extract_profile_func: Function to extract profile using AI
        resume_parser: ResumeParser instance for structured extraction
        ProgressTracker: Progress tracking class
        _websocket_keepalive: WebSocket keepalive function
    """
    with ProgressTracker("Processing your resume", total_steps=4) as tracker:
        try:
            # Step 1: Extract text from file
            tracker.update(1, "📄 Extracting text from resume...")
            _websocket_keepalive("Extracting text")
            
            resume_text = extract_text_func(uploaded_file)
            
            if not resume_text:
                st.error("❌ Could not extract text from resume. Please try a different file format.")
                return
            
            # Store raw text in session state
            st.session_state.resume_text = resume_text
            st.session_state.resume_filename = uploaded_file.name
            
            # Step 2: Extract profile using AI
            tracker.update(2, "🤖 AI is analyzing your profile...")
            _websocket_keepalive("AI profile extraction")
            
            profile_data = extract_profile_func(resume_text)
            
            if not profile_data:
                st.warning("⚠️ Could not extract profile information. Please check your resume format.")
                return
            
            # Step 3: Optional structured extraction for additional validation
            tracker.update(3, "🔍 Validating extracted data...")
            _websocket_keepalive("Validating profile")
            
            # Use ResumeParser for additional structured extraction if needed
            # This provides a second layer of extraction/validation
            structured_profile = resume_parser.extract_structured_profile(
                resume_text, 
                enable_verification=False  # Use single pass for speed
            )
            
            # Merge structured profile data if available
            if structured_profile:
                # Merge without overwriting existing data
                for key, value in structured_profile.items():
                    if key not in profile_data or not profile_data.get(key):
                        profile_data[key] = value
            
            # Step 4: Store in session state
            tracker.update(4, "✅ Profile extraction complete!")
            _websocket_keepalive("Complete")
            
            st.session_state.extracted_profile = profile_data
            st.session_state.user_profile = profile_data  # Also update user_profile for compatibility
            st.session_state.profile_extracted = True
            
            st.success("🎉 Profile extracted successfully!")
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Error processing resume: {str(e)}")


def _display_upload_instructions():
    """Display instructions for resume upload."""
    st.info("📄 **Upload your resume above to get started!**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📋 How it works:
        
        1. **📁 Upload** your resume (PDF, DOCX, or TXT)
        2. **📄 Extract** text from your document
        3. **🤖 AI analyzes** your experience and skills
        4. **✨ Profile** is automatically populated
        5. **🎯 Ready** for job matching and tailored resumes!
        """)
    
    with col2:
        st.markdown("""
        ### 💡 Tips for best results:
        
        - Use **text-based PDFs** (not scanned images)
        - Include **clear section headers**
        - List **skills and technologies** explicitly
        - Add **dates** for work experience
        - Mention **company names** clearly
        """)


def _display_extracted_profile():
    """Display the extracted profile information."""
    profile = st.session_state.extracted_profile
    
    st.markdown("---")
    st.subheader("📋 Extracted Profile Information")
    
    # Basic information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Personal Information**")
        st.write(f"👤 **Name:** {profile.get('name', 'N/A')}")
        st.write(f"📧 **Email:** {profile.get('email', 'N/A')}")
        st.write(f"📱 **Phone:** {profile.get('phone', 'N/A')}")
        st.write(f"📍 **Location:** {profile.get('location', 'N/A')}")
    
    with col2:
        st.markdown("**Professional Links**")
        linkedin = profile.get('linkedin', 'N/A')
        portfolio = profile.get('portfolio', 'N/A')
        st.write(f"🔗 **LinkedIn:** {linkedin}")
        st.write(f"🌐 **Portfolio:** {portfolio}")
    
    # Professional summary
    st.markdown("---")
    st.markdown("**📝 Professional Summary**")
    st.write(profile.get('summary', 'No summary extracted'))
    
    # Skills
    st.markdown("---")
    st.markdown("**💼 Skills**")
    skills = profile.get('skills', '')
    if skills:
        if isinstance(skills, list):
            skills_text = ', '.join(skills)
        else:
            skills_text = skills
        
        # Display as tags
        skills_list = [s.strip() for s in skills_text.split(',') if s.strip()]
        skills_html = ""
        for skill in skills_list[:15]:  # Show first 15
            skills_html += f'<span style="background-color: #E8F4FD; padding: 5px 10px; margin: 3px; border-radius: 5px; display: inline-block;">{skill}</span> '
        
        if skills_html:
            st.markdown(skills_html, unsafe_allow_html=True)
        
        if len(skills_list) > 15:
            with st.expander(f"➕ Show all {len(skills_list)} skills"):
                more_skills_html = ""
                for skill in skills_list[15:]:
                    more_skills_html += f'<span style="background-color: #F0F0F0; padding: 5px 10px; margin: 3px; border-radius: 5px; display: inline-block;">{skill}</span> '
                st.markdown(more_skills_html, unsafe_allow_html=True)
    else:
        st.write("No skills extracted")
    
    # Experience and Education
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("---")
        st.markdown("**📈 Experience**")
        experience = profile.get('experience', 'No experience extracted')
        if len(str(experience)) > 500:
            with st.expander("View full experience"):
                st.write(experience)
        else:
            st.write(experience)
    
    with col2:
        st.markdown("---")
        st.markdown("**🎓 Education**")
        st.write(profile.get('education', 'No education extracted'))
    
    # Certifications
    st.markdown("---")
    st.markdown("**🏆 Certifications & Achievements**")
    st.write(profile.get('certifications', 'No certifications extracted'))
    
    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Go to Job Matching", use_container_width=True):
            st.session_state.current_page = "job_recommendations"
            st.rerun()
    
    with col2:
        if st.button("📝 Generate Tailored Resume", use_container_width=True):
            st.session_state.current_page = "tailored_resume"
            st.rerun()
    
    with col3:
        if st.button("🔄 Upload New Resume", use_container_width=True):
            # Clear extracted profile
            st.session_state.extracted_profile = None
            st.session_state.resume_text = None
            st.session_state.profile_extracted = False
            st.rerun()


# Standalone function for direct text processing (alternative entry point)
def process_resume_text(resume_text: str, enable_verification: bool = False) -> Optional[Dict]:
    """Process resume text directly without file upload.
    
    This is an alternative entry point for when resume text is already available
    (e.g., from another source or pasted text).
    
    Args:
        resume_text: Raw resume text
        enable_verification: Whether to enable two-pass verification
        
    Returns:
        Extracted profile dictionary or None if extraction fails
    """
    from core.resume_parser import ResumeParser, extract_profile_from_resume
    
    if not resume_text or len(resume_text.strip()) < 50:
        return None
    
    # Use modules/resume_upload for primary extraction
    profile_data = extract_profile_from_resume(resume_text)
    
    if not profile_data:
        return None
    
    # Optionally use ResumeParser for additional structured extraction
    if enable_verification:
        resume_parser = ResumeParser()
        structured_profile = resume_parser.extract_structured_profile(
            resume_text, 
            enable_verification=True
        )
        
        if structured_profile:
            for key, value in structured_profile.items():
                if key not in profile_data or not profile_data.get(key):
                    profile_data[key] = value
    
    return profile_data
