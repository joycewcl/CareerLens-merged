"""
Services package for CareerLens application.

This package contains external service clients and integrations.
"""
from .azure_openai import (
    AzureOpenAIClient,
    EmbeddingGenerator,
    TextGenerator,
    generate_docx_from_json,
    generate_pdf_from_json,
    format_resume_as_text,
    set_cell_shading,
    add_horizontal_line,
)
from .linkedin_api import (
    LinkedInJobSearcher,
    get_linkedin_job_searcher
)
from .indeed_api import (
    IndeedJobScraper,
    IndeedScraperAPI,  # Backward compatibility alias
    get_indeed_job_scraper
)

__all__ = [
    # Azure OpenAI
    'AzureOpenAIClient',
    'EmbeddingGenerator',
    'TextGenerator',
    # Resume formatters
    'generate_docx_from_json',
    'generate_pdf_from_json',
    'format_resume_as_text',
    'set_cell_shading',
    'add_horizontal_line',
    # LinkedIn API
    'LinkedInJobSearcher',
    'get_linkedin_job_searcher',
    # Indeed API
    'IndeedJobScraper',
    'IndeedScraperAPI',
    'get_indeed_job_scraper',
]
