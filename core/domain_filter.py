# core/domain_filter.py
"""
Domain/industry filtering functionality for job matching.

This module provides:
- Domain keyword mappings for filtering
- Industry-based job filtering
"""
from typing import List, Dict, Set


# Domain keyword mappings for filtering
DOMAIN_KEYWORDS = {
    'FinTech': ['fintech', 'financial technology', 'blockchain', 'crypto', 'payment', 'banking technology', 'digital banking'],
    'ESG & Sustainability': ['esg', 'sustainability', 'environmental', 'green', 'carbon', 'climate', 'renewable'],
    'Data Analytics': ['data analytics', 'data analysis', 'business intelligence', 'bi', 'data science', 'analytics', 'big data'],
    'Digital Transformation': ['digital transformation', 'digitalization', 'digital strategy', 'innovation'],
    'Investment Banking': ['investment banking', 'ib', 'm&a', 'mergers', 'acquisitions', 'capital markets', 'equity research'],
    'Consulting': ['consulting', 'consultant', 'advisory', 'strategy consulting', 'management consulting'],
    'Technology': ['software', 'technology', 'tech', 'engineering', 'developer', 'programming', 'it'],
    'Healthcare': ['healthcare', 'medical', 'health', 'hospital', 'clinical', 'pharmaceutical', 'biotech'],
    'Education': ['education', 'teaching', 'academic', 'university', 'school', 'e-learning', 'edtech'],
    'Real Estate': ['real estate', 'property', 'realty', 'property management'],
    'Retail & E-commerce': ['retail', 'e-commerce', 'ecommerce', 'online retail'],
    'Marketing & Advertising': ['marketing', 'advertising', 'brand', 'digital marketing', 'social media'],
    'Legal': ['legal', 'law', 'attorney', 'lawyer', 'compliance', 'regulatory'],
    'Human Resources': ['human resources', 'hr', 'recruitment', 'talent acquisition', 'people operations'],
    'Operations': ['operations', 'supply chain', 'logistics', 'procurement']
}


class DomainFilter:
    """Handles industry/domain filtering"""
    
    # Class-level access to domain keywords
    DOMAIN_KEYWORDS = DOMAIN_KEYWORDS
    
    @staticmethod
    def filter_jobs_by_domains(
        jobs: List[Dict],
        target_domains: List[str]
    ) -> List[Dict]:
        """Filter jobs by target domains/industries.
        
        This function filters jobs based on whether they match any of the specified
        industry domains. It searches across job title, description, and company name.
        
        Args:
            jobs: List of job dictionaries
            target_domains: List of domain names to filter by (e.g., ['FinTech', 'Technology'])
            
        Returns:
            Filtered list of jobs matching at least one domain, or all jobs if no matches
        """
        if not target_domains:
            return jobs
        
        filtered = []
        for job in jobs:
            title_lower = job.get('title', '').lower()
            desc_lower = job.get('description', '').lower()
            company_lower = job.get('company', '').lower()
            industry_lower = job.get('industry', '').lower()
            combined = f"{title_lower} {desc_lower} {company_lower} {industry_lower}"
            
            for domain in target_domains:
                keywords = DOMAIN_KEYWORDS.get(domain, [domain.lower()])
                if any(keyword.lower() in combined for keyword in keywords):
                    filtered.append(job)
                    break
        
        return filtered if filtered else jobs
    
    @staticmethod
    def _extract_domain(job: Dict) -> str:
        """Extract domain from job metadata.
        
        Attempts to identify the primary domain/industry of a job based on
        its title, description, company, and industry fields.
        
        Args:
            job: Job dictionary containing job details
            
        Returns:
            The identified domain name, or 'Other' if no match found
        """
        title_lower = job.get('title', '').lower()
        desc_lower = job.get('description', '').lower()
        company_lower = job.get('company', '').lower()
        industry_lower = job.get('industry', '').lower()
        combined = f"{title_lower} {desc_lower} {company_lower} {industry_lower}"
        
        for domain, keywords in DOMAIN_KEYWORDS.items():
            if any(keyword.lower() in combined for keyword in keywords):
                return domain
        
        return 'Other'
    
    @staticmethod
    def get_available_domains() -> List[str]:
        """Get list of all available domain names.
        
        Returns:
            List of domain names that can be used for filtering
        """
        return list(DOMAIN_KEYWORDS.keys())
    
    @staticmethod
    def get_domain_keywords(domain: str) -> List[str]:
        """Get keywords for a specific domain.
        
        Args:
            domain: The domain name to get keywords for
            
        Returns:
            List of keywords associated with the domain, or empty list if not found
        """
        return DOMAIN_KEYWORDS.get(domain, [])


# Backward compatibility alias for function-style imports
def filter_jobs_by_domains(jobs: List[Dict], target_domains: List[str]) -> List[Dict]:
    """Backward compatibility wrapper for DomainFilter.filter_jobs_by_domains"""
    return DomainFilter.filter_jobs_by_domains(jobs, target_domains)
