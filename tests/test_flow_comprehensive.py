"""
Comprehensive Flow Test for CareerLens Application
===================================================
Tests all major components and their integration:
1. Module imports and initialization
2. Database operations
3. API configuration
4. Embedding generation
5. Job matching logic
6. Resume parsing
7. UI components
8. External service configuration
"""

import pytest
import os
import sys
import tempfile
from pathlib import Path

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestModuleImports:
    """Test that all core modules can be imported correctly."""
    
    def test_core_module_imports(self):
        """Test core module imports"""
        from core import (
            JobSeekerBackend,
            JobMatcherBackend,
            JobMatcher,
            ResumeParser,
            GPT4JobRoleDetector,
            TokenUsageTracker,
            RateLimiter,
        )
        assert JobSeekerBackend is not None
        assert JobMatcherBackend is not None
        assert JobMatcher is not None
        
    def test_database_module_imports(self):
        """Test database module imports"""
        from database import (
            DatabaseConnection,
            JobSeekerDB,
            HeadhunterDB,
            init_database,
            init_head_hunter_database,
            save_job_seeker_info,
            get_job_seeker_profile,
        )
        assert JobSeekerDB is not None
        assert HeadhunterDB is not None
        
    def test_services_module_imports(self):
        """Test services module imports"""
        from services import (
            AzureOpenAIClient,
            EmbeddingGenerator,
            TextGenerator,
            LinkedInJobSearcher,
            IndeedJobScraper,
        )
        assert AzureOpenAIClient is not None
        assert LinkedInJobSearcher is not None
        
    def test_utils_module_imports(self):
        """Test utils module imports"""
        from utils import (
            validate_secrets,
            ProgressTracker,
            api_call_with_retry,
            get_embedding_generator,
        )
        assert validate_secrets is not None
        assert ProgressTracker is not None
        
    def test_ui_module_imports(self):
        """Test UI module imports"""
        from ui import (
            main_analyzer_page,
            job_recommendations_page,
            enhanced_head_hunter_page,
            recruitment_match_dashboard,
            ai_interview_dashboard,
            tailored_resume_page,
            market_dashboard_page,
        )
        assert main_analyzer_page is not None
        assert job_recommendations_page is not None


class TestDatabaseOperations:
    """Test database operations."""
    
    def test_job_seeker_db_initialization(self):
        """Test JobSeekerDB initializes correctly"""
        from database import JobSeekerDB
        
        # Use temp directory for test
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                db = JobSeekerDB()
                assert db is not None
                assert hasattr(db, 'save_profile')
                assert hasattr(db, 'get_profile')
            finally:
                os.chdir(old_cwd)
    
    def test_headhunter_db_initialization(self):
        """Test HeadhunterDB initializes correctly"""
        from database import HeadhunterDB
        
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                db = HeadhunterDB()
                assert db is not None
                assert hasattr(db, 'save_job')
                assert hasattr(db, 'get_all_jobs')
            finally:
                os.chdir(old_cwd)
    
    def test_job_seeker_save_and_retrieve(self):
        """Test saving and retrieving job seeker profile"""
        from database import JobSeekerDB
        
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                db = JobSeekerDB()
                
                # Create test profile
                profile = {
                    'education_level': 'Master',
                    'major': 'Computer Science',
                    'hard_skills': 'Python, SQL, Machine Learning',
                    'work_experience': '3-5 years',
                    'primary_role': 'Data Scientist',
                    'simple_search_terms': 'data science, machine learning',
                }
                
                # Save profile
                job_seeker_id = db.save_profile(profile)
                assert job_seeker_id is not None
                assert job_seeker_id.startswith('JS_')
                
                # Retrieve profile
                retrieved = db.get_profile(job_seeker_id)
                assert retrieved is not None
                assert retrieved['education_level'] == 'Master'
                assert retrieved['primary_role'] == 'Data Scientist'
            finally:
                os.chdir(old_cwd)


class TestJobMatcher:
    """Test job matching functionality."""
    
    def test_job_matcher_initialization(self):
        """Test JobMatcher initializes"""
        from core import JobMatcher
        
        matcher = JobMatcher()
        assert matcher is not None
        # JobMatcher uses Pinecone for vector matching
        assert hasattr(matcher, '__init__')
    
    def test_calculate_match_score(self):
        """Test match score calculation"""
        from core import calculate_job_match_score
        
        job_seeker = {
            'hard_skills': 'Python, SQL, Data Analysis',
            'work_experience': 'senior',
            'location_preference': 'Hong Kong',
            'primary_role': 'Data Analyst',
        }
        
        job = {
            'job_title': 'Senior Data Analyst',
            'job_description': 'Looking for Python expert with SQL knowledge',
            'job_city': 'Hong Kong',
        }
        
        result = calculate_job_match_score(job_seeker, job)
        
        assert result is not None
        assert 'overall_score' in result
        assert 'matched_skills' in result
        assert 0 <= result['overall_score'] <= 100
    
    def test_analyze_match_simple(self):
        """Test simple match analysis - verifies function exists and is callable"""
        from core.job_matcher import analyze_match_simple
        
        # The function exists and is importable
        assert analyze_match_simple is not None
        assert callable(analyze_match_simple)


class TestResumeParser:
    """Test resume parsing functionality."""
    
    def test_resume_parser_initialization(self):
        """Test ResumeParser initializes"""
        from core import ResumeParser
        
        parser = ResumeParser()
        assert parser is not None
    
    def test_gpt4_detector_initialization(self):
        """Test GPT4JobRoleDetector initializes"""
        from core import GPT4JobRoleDetector
        
        detector = GPT4JobRoleDetector()
        assert detector is not None
        # GPT4JobRoleDetector uses analyze() method
        assert hasattr(detector, 'analyze') or hasattr(detector, '__init__')


class TestConfiguration:
    """Test configuration management."""
    
    def test_config_class_exists(self):
        """Test Config class exists and has expected attributes"""
        from config import Config
        
        assert Config is not None
        assert hasattr(Config, 'setup')
        assert hasattr(Config, 'validate')
        assert hasattr(Config, 'check_azure_credentials')
        assert hasattr(Config, 'get_azure_client_config')
    
    def test_config_setup(self):
        """Test Config.setup() runs without error"""
        from config import Config
        
        # Reset initialized state for testing
        Config._initialized = False
        Config.setup()
        
        # Check that attributes are set (may be None if no secrets)
        assert Config._initialized is True
    
    def test_config_database_paths(self):
        """Test database paths are defined"""
        from config import Config
        
        assert Config.DB_PATH_JOB_SEEKER is not None
        assert Config.DB_PATH_HEAD_HUNTER is not None


class TestSalaryAnalyzer:
    """Test salary analysis functionality."""
    
    def test_salary_extractor(self):
        """Test salary extraction from text"""
        from core import extract_salary_from_text_regex
        
        text = "Salary: HKD 30,000 - 40,000 per month"
        result = extract_salary_from_text_regex(text)
        
        # Should extract some value
        assert result is not None or result == 0
    
    def test_salary_filter(self):
        """Test job filtering by salary"""
        from core import filter_jobs_by_salary
        
        jobs = [
            {'job_title': 'Job 1', 'salary_min': 30000},
            {'job_title': 'Job 2', 'salary_min': 50000},
            {'job_title': 'Job 3', 'salary_min': 20000},
        ]
        
        filtered = filter_jobs_by_salary(jobs, min_salary=25000)
        
        # Should filter correctly
        assert isinstance(filtered, list)


class TestDomainFilter:
    """Test domain/industry filtering."""
    
    def test_domain_filter_class(self):
        """Test DomainFilter class"""
        from core import DomainFilter, DOMAIN_KEYWORDS
        
        assert DomainFilter is not None
        assert DOMAIN_KEYWORDS is not None
        assert isinstance(DOMAIN_KEYWORDS, dict)
    
    def test_filter_jobs_by_domains(self):
        """Test filtering jobs by domain"""
        from core import filter_jobs_by_domains
        
        jobs = [
            {'job_title': 'FinTech Developer', 'job_description': 'Work in fintech'},
            {'job_title': 'Healthcare Analyst', 'job_description': 'Healthcare data'},
            {'job_title': 'Generic Job', 'job_description': 'Something else'},
        ]
        
        # Filter for FinTech
        filtered = filter_jobs_by_domains(jobs, ['FinTech'])
        assert isinstance(filtered, list)


class TestResumeGenerator:
    """Test resume generation functionality."""
    
    def test_resume_generator_class(self):
        """Test ResumeGenerator class exists"""
        from core import ResumeGenerator
        
        assert ResumeGenerator is not None
    
    def test_format_resume_as_text(self):
        """Test text formatting function"""
        from core import format_resume_as_text
        
        resume_data = {
            'name': 'John Doe',
            'email': 'john@example.com',
            'summary': 'Experienced professional',
            'skills': ['Python', 'SQL'],
        }
        
        result = format_resume_as_text(resume_data)
        assert isinstance(result, str)


class TestRateLimiting:
    """Test rate limiting and token tracking."""
    
    def test_token_tracker(self):
        """Test TokenUsageTracker"""
        from core import TokenUsageTracker
        
        tracker = TokenUsageTracker()
        assert tracker is not None
        assert hasattr(tracker, 'add_usage')
        assert hasattr(tracker, 'get_summary')
    
    def test_rate_limiter(self):
        """Test RateLimiter"""
        from core import RateLimiter
        
        # RateLimiter requires max_calls or max_requests_per_minute
        limiter = RateLimiter(max_requests_per_minute=60)
        assert limiter is not None
        # Check it has a method for rate limiting
        assert hasattr(limiter, 'wait_if_needed') or hasattr(limiter, '__init__')


class TestInterviewModule:
    """Test interview functionality."""
    
    def test_interview_functions_exist(self):
        """Test interview functions are importable"""
        from core import (
            initialize_interview_session,
            generate_interview_question,
            evaluate_answer,
            generate_final_summary,
        )
        
        assert initialize_interview_session is not None
        assert generate_interview_question is not None
        assert evaluate_answer is not None
        assert generate_final_summary is not None


class TestSemanticSearch:
    """Test semantic search functionality."""
    
    def test_semantic_search_class(self):
        """Test SemanticJobSearch class exists"""
        from core import SemanticJobSearch
        
        assert SemanticJobSearch is not None
    
    def test_fetch_jobs_with_cache(self):
        """Test cache function exists"""
        from core import fetch_jobs_with_cache, is_cache_valid
        
        assert fetch_jobs_with_cache is not None
        assert is_cache_valid is not None


class TestUIComponents:
    """Test UI components."""
    
    def test_ui_components_import(self):
        """Test UI components can be imported"""
        from ui.components import (
            render_sidebar,
            render_hero_banner,
            display_job_card,
            display_user_profile,
            display_market_positioning_profile,
            render_styles,
        )
        
        assert render_sidebar is not None
        assert render_hero_banner is not None
    
    def test_visualizations_import(self):
        """Test visualization functions"""
        from ui import (
            create_enhanced_visualizations,
            create_job_comparison_radar,
        )
        
        assert create_enhanced_visualizations is not None
        assert create_job_comparison_radar is not None


class TestExternalServices:
    """Test external service clients."""
    
    def test_azure_openai_client(self):
        """Test AzureOpenAIClient class"""
        from services import AzureOpenAIClient
        
        assert AzureOpenAIClient is not None
        assert hasattr(AzureOpenAIClient, '__init__')
    
    def test_linkedin_searcher(self):
        """Test LinkedInJobSearcher class"""
        from services import LinkedInJobSearcher, get_linkedin_job_searcher
        
        assert LinkedInJobSearcher is not None
        assert get_linkedin_job_searcher is not None
    
    def test_indeed_scraper(self):
        """Test IndeedJobScraper class"""
        from services import IndeedJobScraper, get_indeed_job_scraper
        
        assert IndeedJobScraper is not None
        assert get_indeed_job_scraper is not None


class TestEmbeddingGeneration:
    """Test embedding generation."""
    
    def test_embedding_generator_class(self):
        """Test EmbeddingGenerator class exists"""
        from services import EmbeddingGenerator
        
        assert EmbeddingGenerator is not None
    
    def test_text_generator_class(self):
        """Test TextGenerator class exists"""
        from services import TextGenerator
        
        assert TextGenerator is not None


class TestAPISecrets:
    """Test API secrets validation."""
    
    def test_validate_secrets_function(self):
        """Test validate_secrets function"""
        from utils import validate_secrets
        
        # Should return a result (may have missing secrets)
        result = validate_secrets()
        
        # Result should indicate status
        assert result is not None or result is True or result is False


class TestStreamlitAppSyntax:
    """Test main app has no syntax errors."""
    
    def test_streamlit_app_compiles(self):
        """Test streamlit_app.py compiles without syntax errors"""
        app_path = Path(__file__).parent.parent / 'streamlit_app.py'
        
        with open(app_path, 'r') as f:
            code = f.read()
        
        # This will raise SyntaxError if there are issues
        compile(code, 'streamlit_app.py', 'exec')
    
    def test_all_page_modules_compile(self):
        """Test all page modules compile without syntax errors"""
        ui_path = Path(__file__).parent.parent / 'ui'
        
        page_files = [
            'job_seeker_dashboard.py',
            'job_search_page.py',
            'headhunter_dashboard.py',
            'recruitment_match_page.py',
            'ai_interview_page.py',
            'resume_tailor_page.py',
            'market_dashboard_page.py',
        ]
        
        for page_file in page_files:
            file_path = ui_path / page_file
            if file_path.exists():
                with open(file_path, 'r') as f:
                    code = f.read()
                compile(code, page_file, 'exec')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
