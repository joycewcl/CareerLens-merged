# tests/test_backward_compatibility.py
"""
Test that module imports work from their canonical locations.
"""

def test_core_imports_work():
    """Ensure core module imports work"""
    
    from core import (
        analyze_match_simple,
        JobSeekerBackend,
        JobMatcherBackend,
        extract_structured_profile,
        generate_tailored_resume
    )
    
    assert analyze_match_simple is not None
    assert JobSeekerBackend is not None


def test_services_imports_work():
    """Ensure services module imports work"""
    
    from services.linkedin_api import LinkedInJobSearcher
    from services.azure_openai import (
        generate_docx_from_json,
        generate_pdf_from_json,
        format_resume_as_text
    )
    
    assert LinkedInJobSearcher is not None
    assert generate_docx_from_json is not None


def test_database_imports_work():
    """Ensure database module imports work"""
    
    from database.queries import (
        get_all_job_seekers,
        get_job_seeker_profile,
        save_job_seeker_info,
        get_all_jobs_for_matching
    )
    
    assert get_all_job_seekers is not None
    assert save_job_seeker_info is not None


def test_analysis_imports_work():
    """Ensure analysis module imports work from core"""
    
    from core.salary_analyzer import (
        SalaryAnalyzer,
        filter_jobs_by_salary,
        calculate_salary_band
    )
    from core.domain_filter import (
        DomainFilter,
        filter_jobs_by_domains
    )
    
    # Test class imports
    assert SalaryAnalyzer is not None
    assert DomainFilter is not None
    
    # Test backward compatibility function imports
    assert filter_jobs_by_domains is not None
    assert filter_jobs_by_salary is not None
    assert calculate_salary_band is not None


def test_imports_are_same_object():
    """Verify imports from different paths point to same objects"""
    
    from services.linkedin_api import LinkedInJobSearcher as DirectImport
    from services import LinkedInJobSearcher as PackageImport
    
    # Should be the SAME class object
    assert DirectImport is PackageImport
    
    from core.job_matcher import JobMatcher as DirectMatcher
    from core import JobMatcher as PackageImport
    
    assert DirectMatcher is PackageImport
