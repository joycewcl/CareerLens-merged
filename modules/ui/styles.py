"""CSS styles and JavaScript for CareerLens UI"""
import json
import os
import textwrap

import streamlit as st
import streamlit.components.v1 as components

# Lazy load logo - only when needed
_logo_html = None
_logo_loaded = False


def _load_logo():
    """Lazy load logo for hero banner"""
    global _logo_html, _logo_loaded
    if _logo_loaded:
        return _logo_html
    
    _logo_loaded = True
    from modules.utils.helpers import get_img_as_base64
    
    logo_paths = ["logo.png", "CareerLens_Logo.png"]
    for logo_path in logo_paths:
        if os.path.exists(logo_path):
            try:
                logo_base64 = get_img_as_base64(logo_path)
                _logo_html = f'<img src="data:image/png;base64,{logo_base64}" class="hero-bg-logo">'
                return _logo_html
            except Exception:
                _logo_html = '<div class="hero-bg-logo"></div>'
                return _logo_html
    
    _logo_html = '<div class="hero-bg-logo"></div>'
    return _logo_html


def _inject_global_js(js_code: str, script_id: str) -> None:
    """Inject a JS snippet into the parent Streamlit document exactly once."""
    cleaned_js = textwrap.dedent(js_code).strip()
    if not cleaned_js:
        return

    # Use a lightweight HTML component to append the script to the parent DOM.
    components.html(
        f"""
        <script>
        (function() {{
            let doc = document;
            try {{
                if (window.parent && window.parent.document) {{
                    doc = window.parent.document;
                }}
            }} catch (err) {{
                console.warn('CareerLens: Unable to access parent document for script injection.', err);
            }}

            if (doc.getElementById('{script_id}')) {{
                return;
            }}

            const script = doc.createElement('script');
            script.id = '{script_id}';
            script.type = 'text/javascript';
            script.innerHTML = {json.dumps(cleaned_js)};
            doc.body.appendChild(script);
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def render_styles():
    """Render all CSS styles and JavaScript for the application"""
    st.markdown("""
    <style>
        /* Import Google Fonts for CareerLens branding */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Montserrat:wght@400;700&display=swap');
        
        /* CareerLens Design System - CSS Variables */
        :root {{
            /* Backgrounds */
            --bg-primary: #0F172A;   /* Deep Midnight Navy */
            --bg-secondary: #1E293B; /* Slightly lighter navy for cards/sections */

            /* Text */
            --text-primary-light: #FFFFFF;
            --text-secondary: #94A3B8; /* Muted grey-blue for subtitles */

            /* Brand Colors */
            --brand-glow: #00D2FF;   /* The bright cyan highlight */
            --brand-core: #0084C2;   /* The standard logo blue */
            
            /* UI Elements */
            --accent-gradient: linear-gradient(135deg, #00D2FF 0%, #0084C2 100%);
            
            /* Legacy aliases for backwards compatibility */
            --navy: var(--bg-primary);
            --cyan: var(--brand-glow);
            --primary-accent: var(--brand-core);
            --action-accent: var(--brand-glow);
            
            /* UI Colors */
            --bg-gray: #f3f4f6;
            --bg-main: #f3f4f6;
            --bg-container: #F4F7FC;
            --card-bg: #FFFFFF;
            --text-primary: #161616;
            --border-color: #E0E0E0;
            --hover-bg: #F0F0F0;
            --success-green: #10B981;
            --warning-amber: #F59E0B;
            --error-red: #EF4444;
            --navy-deep: #1e3a5f;
            --navy-light: #2C3E50;
            --btn-text: #0a0a0a;
        }}
        
        [data-theme="dark"],
        html[data-theme="dark"],
        html[data-theme="dark"] :root {{
            --primary-accent: #4589FF;
            --action-accent: #4589FF;
            --bg-main: #161616;
            --bg-container: #262626;
            --card-bg: #262626;
            --text-primary: #F4F4F4;
            --border-color: #3D3D3D;
            --hover-bg: #333333;
            --navy: #1e293b;
            --cyan: #22d3ee;
            --bg-gray: #1f2937;
        }}
        
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header[data-testid="stHeader"] {{visibility: hidden; height: 0; padding: 0; margin: 0;}}
        .stDeployButton {{display: none;}}
        
        .stApp {{
            background-color: var(--bg-gray);
            color: var(--text-primary);
        }}
        
        [data-testid="stSidebar"] {{
            background-color: var(--bg-primary);
            padding: 2rem 1rem;
        }}
        [data-testid="stSidebar"] * {{
            color: var(--text-secondary) !important;
            font-family: 'Inter', sans-serif;
        }}
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] .stMarkdown h2,
        [data-testid="stSidebar"] .stMarkdown h3 {{
            color: var(--text-primary-light) !important;
            font-family: 'Montserrat', sans-serif;
        }}
        [data-testid="stSidebar"] .stButton > button {{
            background: var(--accent-gradient);
            color: #FFFFFF !important;
            font-family: 'Montserrat', sans-serif;
            font-weight: 700 !important;
            border: none;
            border-radius: 50px;
            box-shadow: 0 0 20px rgba(0, 210, 255, 0.4);
        }}
        [data-testid="stSidebar"] .stButton > button:hover {{
            box-shadow: 0 0 30px rgba(0, 210, 255, 0.6);
            transform: translateY(-2px);
        }}
        
        .hero-container {{
            background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
            padding: 40px;
            border-radius: 12px;
            color: white;
            position: relative;
            overflow: hidden;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            font-size: 0;
        }}
        .hero-container > * {{
            font-size: 16px;
        }}
        .hero-content {{
            position: relative;
            z-index: 10;
        }}
        .hero-title {{
            font-size: 32px;
            font-weight: 700;
            margin: 0;
            color: white;
        }}
        .hero-subtitle {{
            color: var(--text-secondary);
            font-size: 16px;
            margin-top: 10px;
        }}
        .hero-bg-logo {{
            position: absolute;
            right: -30px;
            top: -30px;
            width: 250px;
            opacity: 0.15;
            transform: rotate(-15deg);
            pointer-events: none;
            z-index: 5;
        }}
        
        .dashboard-metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .dashboard-metric-label {{
            font-size: 12px;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .dashboard-metric-value {{
            font-size: 28px;
            font-weight: 700;
            color: #111827;
            margin-top: 5px;
        }}
        
        [data-theme="dark"] .hero-container {{
            background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
        }}
        [data-theme="dark"] .dashboard-metric-card {{
            background: var(--card-bg);
        }}
        [data-theme="dark"] .dashboard-metric-value {{
            color: var(--text-primary);
        }}
        
        .job-card {{
            background-color: var(--bg-container);
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            border: none;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .job-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        }}
        
        .match-score {{
            background: var(--accent-gradient);
            color: white;
            padding: 0.4rem 1rem;
            border-radius: 20px;
            font-weight: bold;
            display: inline-block;
            font-size: 0.9rem;
        }}
        
        .tag {{
            display: inline-block;
            background-color: var(--bg-container);
            color: var(--text-primary);
            padding: 0.3rem 0.8rem;
            border-radius: 12px;
            margin: 0.2rem;
            font-size: 0.85rem;
            border: none;
        }}
        
        .match-score-display {{
            font-size: 2rem;
            font-weight: bold;
            color: var(--brand-glow);
            text-align: center;
        }}
        
        .main-header {{
            font-size: 3rem;
            font-weight: bold;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center;
            margin-bottom: 1rem;
            letter-spacing: -0.02em;
        }}
        
        .ws-reconnecting-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.7);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 99999;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.3s, visibility 0.3s;
        }}
        .ws-reconnecting-overlay.active {{
            opacity: 1;
            visibility: visible;
        }}
        .ws-reconnecting-content {{
            background: white;
            padding: 30px 40px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }}
        [data-theme="dark"] .ws-reconnecting-content {{
            background: #262626;
            color: #f4f4f4;
        }}
        .ws-reconnecting-spinner {{
            width: 40px;
            height: 40px;
            border: 4px solid #e0e0e0;
            border-top-color: #0F62FE;
            border-radius: 50%;
            animation: ws-spin 1s linear infinite;
            margin: 0 auto 15px;
        }}
        @keyframes ws-spin {{
            to {{ transform: rotate(360deg); }}
        }}
        .ws-reconnecting-text {{
            font-size: 16px;
            color: #333;
            margin-bottom: 5px;
        }}
        [data-theme="dark"] .ws-reconnecting-text {{
            color: #f4f4f4;
        }}
        .ws-reconnecting-subtext {{
            font-size: 13px;
            color: #666;
        }}
        [data-theme="dark"] .ws-reconnecting-subtext {{
            color: #999;
        }}
        
        /* How It Works Page Styles */
        .how-it-works-section {{
            background: var(--card-bg);
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
            border-left: 4px solid var(--brand-glow);
        }}
        
        .step-box {{
            background: var(--card-bg);
            padding: 2rem;
            border-radius: 12px;
            margin: 1.5rem 0;
            border: 2px solid var(--brand-core);
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        }}
        
        .step-box h3 {{
            color: var(--brand-glow);
            margin-top: 0;
        }}
        
        .step-box h4 {{
            color: var(--brand-core);
            margin-top: 1.5rem;
        }}
        
        .step-box code {{
            background: var(--bg-container);
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
        }}
        
        .step-box pre {{
            background: var(--bg-container);
            padding: 1rem;
            border-radius: 8px;
            overflow-x: auto;
            border-left: 3px solid var(--brand-glow);
        }}
        
        .tech-badge {{
            display: inline-block;
            background: var(--accent-gradient);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            margin: 0.3rem;
            font-size: 0.9rem;
            font-weight: 600;
        }}
        
        [data-theme="dark"] .how-it-works-section {{
            background: var(--bg-container);
        }}
        
        [data-theme="dark"] .step-box {{
            background: var(--bg-container);
        }}
        
        /* Info Banner */
        .info-banner {{
            background: var(--accent-gradient);
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            color: white;
        }}
        
        .info-banner h3 {{
            margin: 0;
            color: white;
        }}
        
        .info-banner p {{
            margin: 0.5rem 0 0 0;
            opacity: 0.9;
        }}
    </style>
    <div id="ws-reconnecting-overlay" class="ws-reconnecting-overlay">
        <div class="ws-reconnecting-content">
            <div class="ws-reconnecting-spinner"></div>
            <div class="ws-reconnecting-text">Reconnecting...</div>
            <div class="ws-reconnecting-subtext">Please wait while we restore your connection</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _inject_global_js(_STREAMLIT_THEME_AND_RECONNECT_JS, "careerlens-streamlit-reconnect-js")


def get_logo_html():
    """Get logo HTML for hero banner (lazy loaded)"""
    return _load_logo()


_STREAMLIT_THEME_AND_RECONNECT_JS = """
(function() {
    if (window.__careerlensThemeInit__) {
        return;
    }
    window.__careerlensThemeInit__ = true;

    function updateTheme() {
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        const stApp = document.querySelector('.stApp') || document.querySelector('[data-testid="stApp"]');

        if (prefersDark) {
            document.documentElement.setAttribute('data-theme', 'dark');
            document.body.setAttribute('data-theme', 'dark');
            if (stApp) {
                stApp.setAttribute('data-theme', 'dark');
            }
        } else {
            document.documentElement.removeAttribute('data-theme');
            document.body.removeAttribute('data-theme');
            if (stApp) {
                stApp.removeAttribute('data-theme');
            }
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', updateTheme);
    } else {
        updateTheme();
    }

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    if (mediaQuery.addEventListener) {
        mediaQuery.addEventListener('change', updateTheme);
    } else if (mediaQuery.addListener) {
        mediaQuery.addListener(updateTheme);
    }
})();

(function() {
    if (window.__careerlensReconnectInit__) {
        return;
    }
    window.__careerlensReconnectInit__ = true;

    let isReconnecting = false;

    function getOverlay() {
        return document.getElementById('ws-reconnecting-overlay');
    }

    function showReconnectingOverlay() {
        const overlay = getOverlay();
        if (overlay && !isReconnecting) {
            isReconnecting = true;
            overlay.classList.add('active');
        }
    }

    function hideReconnectingOverlay() {
        const overlay = getOverlay();
        if (overlay) {
            isReconnecting = false;
            overlay.classList.remove('active');
        }
    }

    function initReconnectionHandlers() {
        window.addEventListener('offline', function() {
            showReconnectingOverlay();
        });

        window.addEventListener('online', function() {
            setTimeout(function() {
                hideReconnectingOverlay();
            }, 1000);
        });

        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.addedNodes.length) {
                    mutation.addedNodes.forEach(function(node) {
                        if (node.nodeType === 1 && node.textContent && node.textContent.includes('Connecting')) {
                            showReconnectingOverlay();
                        }
                    });
                }
            });
        });

        if (document.body) {
            observer.observe(document.body, {
                childList: true,
                subtree: true
            });
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initReconnectionHandlers);
    } else {
        initReconnectionHandlers();
    }
})();
"""
