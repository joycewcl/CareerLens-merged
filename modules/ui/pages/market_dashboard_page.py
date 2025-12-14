"""
Market Dashboard Page - Comprehensive Market Insights.

This module contains the modular CareerLens dashboard for viewing:
- Market positioning profile
- Ranked job matches
- Match breakdown and analysis
- Application copilot features
"""

import streamlit as st


def market_dashboard_page():
    """Market Dashboard Page - Modular CareerLens Dashboard"""
    # Check if modules are available
    try:
        from modules.utils import _cleanup_session_state, validate_secrets
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
    except ImportError as e:
        MODULES_AVAILABLE = False
        import_error = str(e)
    
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
            st.session_state.get('user_profile', {}),
            st.session_state.matched_jobs if st.session_state.get('dashboard_ready', False) else None
        )
        
        # Main dashboard area - only show after analysis
        if not st.session_state.get('dashboard_ready', False) or not st.session_state.get('matched_jobs'):
            st.info("👆 Upload your CV in the sidebar to get started. Once uploaded, use the 'Refine Results' section below to search for jobs and see your market positioning.")
            
            # Show the Refine Results section even before search to allow user to initiate search
            display_refine_results_section([], st.session_state.get('user_profile', {}))
            return
        
        # Display Market Positioning Profile (Top Section)
        display_market_positioning_profile(
            st.session_state.matched_jobs,
            st.session_state.get('user_profile', {})
        )
        
        # Display Refine Results Section
        display_refine_results_section(
            st.session_state.matched_jobs,
            st.session_state.get('user_profile', {})
        )
        
        # Display Smart Ranked Matches Table (Middle Section)
        display_ranked_matches_table(
            st.session_state.matched_jobs,
            st.session_state.get('user_profile', {})
        )
        
        # Display Match Breakdown & Application Copilot (Bottom Section)
        display_match_breakdown(
            st.session_state.matched_jobs,
            st.session_state.get('user_profile', {})
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
