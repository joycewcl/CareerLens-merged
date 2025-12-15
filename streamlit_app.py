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
from typing import List, Dict

# Import backend utilities
from core.rate_limiting import TokenUsageTracker

# Import How It Works page
from how_it_works import render_how_it_works_page

# Import database initialization
from database import init_database, init_head_hunter_database

# Import configuration
from config import Config

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

# ============================================================================
# IMPORTS FOR PAGE MODULES AND UTILITIES
# ============================================================================
try:
    from utils import _cleanup_session_state, validate_secrets
    from utils.helpers import (
        _chunked_sleep,
        _websocket_keepalive,
        _ensure_websocket_alive,
        ProgressTracker
    )
    from ui.components.styles import render_styles
    from ui import (
        # Page modules
        main_analyzer_page,
        job_recommendations_page,
        enhanced_head_hunter_page,
        recruitment_match_dashboard,
        ai_interview_dashboard,
        tailored_resume_page,
        market_dashboard_page,
        # UI components (for compatibility)
        render_sidebar as modular_render_sidebar,
        render_hero_banner,
        display_resume_generator as modular_display_resume_generator,
        display_market_positioning_profile,
    )
    MODULES_AVAILABLE = True
    WEBSOCKET_UTILS_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    WEBSOCKET_UTILS_AVAILABLE = False
    
    # Provide fallback implementations for WebSocket utilities
    def _websocket_keepalive(message=None, force=False):
        """Fallback no-op implementation"""
        pass
    
    def _ensure_websocket_alive():
        """Fallback no-op implementation"""
        pass
    
    def _chunked_sleep(delay, message_prefix=""):
        """Fallback implementation using regular sleep"""
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
    
    # Import page modules directly if modular UI failed
    try:
        from ui import (
            main_analyzer_page,
            job_recommendations_page,
            enhanced_head_hunter_page,
            recruitment_match_dashboard,
            ai_interview_dashboard,
            tailored_resume_page,
            market_dashboard_page,
        )
        MODULES_AVAILABLE = True
    except ImportError:
        MODULES_AVAILABLE = False


# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================
@st.cache_resource
def initialize_databases():
    init_database()
    init_head_hunter_database()
    return True

_db_initialized = initialize_databases()


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
# Initialize token tracker in session state
if 'token_tracker' not in st.session_state:
    st.session_state.token_tracker = TokenUsageTracker()

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
if time.time() - st.session_state.get('ws_last_activity', time.time()) > 30:
    st.session_state.ws_reconnect_count += 1
    st.session_state.ws_connection_time = time.time()

# Limit search history size
MAX_SEARCH_HISTORY = 5
if len(st.session_state.search_history) > MAX_SEARCH_HISTORY:
    st.session_state.search_history = st.session_state.search_history[:MAX_SEARCH_HISTORY]

# Run memory cleanup after session state is initialized (if modules available)
if MODULES_AVAILABLE and WEBSOCKET_UTILS_AVAILABLE:
    try:
        _cleanup_session_state()
    except Exception:
        pass

# Periodic WebSocket keepalive on app rerun
try:
    _ensure_websocket_alive()
except Exception:
    pass


# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.markdown("""
<style>
    .careerlens-tagline {
        font-family: 'Montserrat', sans-serif;
        color: var(--text-secondary-light);
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 0.7rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Navigation Section Headers */
    .nav-section-header {
        font-family: 'Montserrat', sans-serif;
        font-weight: 700;
        font-size: 1.1rem;
        color: white !important;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        padding-left: 0.5rem;
        border-left: 3px solid var(--brand-glow);
    }
    
    /* Navigation Items */
    .nav-item {
        font-family: 'Inter', sans-serif;
        color: var(--text-secondary-light) !important;
        font-size: 0.9rem;
        padding-left: 1.5rem;
        margin: 0.3rem 0;
        cursor: pointer;
    }
    .nav-item:hover {
        color: var(--brand-glow) !important;
    }
</style>
""", unsafe_allow_html=True)

# Display the CareerLens logo
st.sidebar.image("CareerLens_Logo.png", use_container_width=True)
st.sidebar.markdown('<div class="careerlens-tagline">AI Career Copilot • Hong Kong</div>', unsafe_allow_html=True)

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


# ============================================================================
# PAGE ROUTING
# ============================================================================
if not MODULES_AVAILABLE:
    st.error("❌ Page modules not available. Please ensure the modules/ui/pages directory is properly installed.")
    st.info("Falling back to basic functionality...")
    st.stop()

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


# ============================================================================
# SIDEBAR HELP AND FOOTER
# ============================================================================
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
st.caption("🤖 Powered by Azure OpenAI, Pinecone Vector Search, RapidAPI LinkedIn Jobs, and CareerLens AI")

# Application startup
if __name__ == "__main__":
    # Ensure application runs normally
    pass
