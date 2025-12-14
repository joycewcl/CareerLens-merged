# core/salary_analyzer.py
"""
Salary extraction, filtering, and analysis functionality.

This module provides:
- Salary extraction from text (regex and LLM-based)
- Salary filtering for job lists
- Salary band calculation
"""
import re
import json
from typing import List, Dict, Optional, Tuple


class SalaryAnalyzer:
    """Handles salary filtering and analysis"""
    
    @staticmethod
    def extract_salary_from_text_regex(text: str) -> Tuple[Optional[int], Optional[int]]:
        """Fallback regex-based salary extraction.
        
        Attempts to extract salary information using regular expression patterns.
        
        Args:
            text: Text containing potential salary information
            
        Returns:
            Tuple of (min_salary, max_salary) in HKD or (None, None) if not found
        """
        if not text:
            return None, None
        
        patterns = [
            r'HKD\s*\$?\s*(\d{1,3}(?:,\d{3})*(?:k|K)?)\s*[-–—]\s*\$?\s*(\d{1,3}(?:,\d{3})*(?:k|K)?)',
            r'(\d{1,3}(?:,\d{3})*(?:k|K)?)\s*[-–—]\s*(\d{1,3}(?:,\d{3})*(?:k|K)?)\s*HKD',
            r'HKD\s*\$?\s*(\d{1,3}(?:,\d{3})*(?:k|K)?)\s*(?:per month|/month|/mth|monthly)',
            r'(\d{1,3}(?:,\d{3})*(?:k|K)?)\s*HKD\s*(?:per month|/month|/mth|monthly)',
            r'\$\s*(\d{1,3}(?:,\d{3})*(?:k|K)?)\s*[-–—]\s*\$?\s*(\d{1,3}(?:,\d{3})*(?:k|K)?)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                match = matches[0]
                if isinstance(match, tuple) and len(match) == 2:
                    min_sal = match[0].replace(',', '').replace('k', '000').replace('K', '000')
                    max_sal = match[1].replace(',', '').replace('k', '000').replace('K', '000')
                    try:
                        return int(min_sal), int(max_sal)
                    except (ValueError, TypeError):
                        pass
        
        return None, None
    
    @staticmethod
    def extract_salary_from_text(text: str, config=None) -> Tuple[Optional[int], Optional[int]]:
        """Extract salary information from job description using LLM.
        
        This function first attempts to use Azure OpenAI to extract and normalize
        salary information, then falls back to regex extraction if that fails.
        
        Args:
            text: Text containing potential salary information
            config: Optional config object for Azure credentials
            
        Returns:
            Tuple of (min_salary, max_salary) in monthly HKD or (None, None) if not found
        """
        if not text:
            return None, None
        
        text_for_extraction = text[:3000] if len(text) > 3000 else text
        
        try:
            if config is None:
                from config import Config
                config = Config
            
            # Check if API keys are configured - if not, fall back to regex
            is_configured, _ = config.check_azure_credentials()
            if not is_configured:
                return SalaryAnalyzer.extract_salary_from_text_regex(text)
            
            from openai import AzureOpenAI
            client = AzureOpenAI(
                azure_endpoint=config.AZURE_ENDPOINT,
                api_key=config.AZURE_API_KEY,
                api_version=config.AZURE_API_VERSION
            )
            
            prompt = f"""Extract salary information from this job description text. 
Look for salary ranges, amounts, and compensation details. Normalize everything to monthly HKD (Hong Kong Dollars).

JOB DESCRIPTION TEXT:
{text_for_extraction}

Extract and return salary information as JSON with this structure:
{{
    "min_salary_hkd_monthly": <number or null>,
    "max_salary_hkd_monthly": <number or null>,
    "found": true/false,
    "raw_text": "the exact salary text found in the description"
}}

Rules:
- Convert all amounts to monthly HKD (multiply annual by 12, weekly by 4.33, daily by 22)
- If only one amount is found, set both min and max to that value
- If no salary is found, set "found": false and return null for min/max
- Always return valid JSON, no extra explanation"""

            response = client.chat.completions.create(
                model=config.AZURE_MODEL,
                messages=[
                    {"role": "system", "content": "You are a salary extraction expert. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            salary_data = json.loads(content)
            
            if salary_data.get('found', False):
                min_sal = salary_data.get('min_salary_hkd_monthly')
                max_sal = salary_data.get('max_salary_hkd_monthly')
                if min_sal is not None and max_sal is not None:
                    return int(min_sal), int(max_sal)
                elif min_sal is not None:
                    return int(min_sal), int(min_sal * 1.2)
            
            return SalaryAnalyzer.extract_salary_from_text_regex(text)
            
        except Exception:
            return SalaryAnalyzer.extract_salary_from_text_regex(text)
    
    @staticmethod
    def filter_jobs_by_salary(
        jobs: List[Dict],
        min_salary: Optional[float] = None,
        expected_salary: Optional[float] = None,
        tolerance: float = 0.2
    ) -> List[Dict]:
        """Filter jobs by minimum salary expectation.
        
        This function supports two filtering modes:
        1. If min_salary is provided: Jobs must meet or exceed min_salary
        2. If expected_salary is provided: Jobs must be within tolerance range
        
        Args:
            jobs: List of job dictionaries
            min_salary: Minimum acceptable salary (absolute filter)
            expected_salary: Expected salary (with tolerance filter)
            tolerance: Tolerance range for expected_salary filtering (default 20%)
            
        Returns:
            Filtered list of jobs meeting salary criteria
        """
        # Handle expected_salary mode (with tolerance)
        if expected_salary is not None and expected_salary > 0 and min_salary is None:
            min_acceptable = expected_salary * (1 - tolerance)
            max_acceptable = expected_salary * (1 + tolerance)
            
            filtered = []
            for job in jobs:
                job_min_salary = job.get('min_salary', 0)
                job_max_salary = job.get('max_salary', 0)
                
                if job_min_salary <= max_acceptable and job_max_salary >= min_acceptable:
                    filtered.append(job)
            
            return filtered
        
        # Handle min_salary mode (absolute minimum)
        if not min_salary or min_salary <= 0:
            return jobs
        
        filtered = []
        jobs_without_salary = []
        
        for job in jobs:
            salary_str = job.get('salary', '')
            description = job.get('description', '')
            
            job_min_sal, job_max_sal = SalaryAnalyzer.extract_salary_from_text_regex(salary_str)
            
            if not job_min_sal:
                job_min_sal, job_max_sal = SalaryAnalyzer.extract_salary_from_text_regex(description)
            
            if job_min_sal:
                if job_min_sal >= min_salary or (job_max_sal and job_max_sal >= min_salary):
                    filtered.append(job)
            else:
                jobs_without_salary.append(job)
        
        if filtered:
            return filtered
        elif jobs_without_salary:
            return jobs_without_salary
        else:
            return []
    
    @staticmethod
    def calculate_salary_band(
        matched_jobs: List[Dict] = None,
        expected: float = None
    ) -> Dict[str, float]:
        """Calculate estimated salary band from matched jobs or expected salary.
        
        Args:
            matched_jobs: List of matched job results (optional)
            expected: Expected salary value (optional)
            
        Returns:
            Dictionary with 'min', 'max', and 'ideal' salary values
        """
        # Simple mode: calculate from expected
        if expected is not None and expected > 0:
            return {
                'min': expected * 0.8,
                'max': expected * 1.2,
                'ideal': expected
            }
        
        # Complex mode: calculate from matched jobs
        if matched_jobs is None:
            return {'min': 45000, 'max': 55000, 'ideal': 50000}
        
        salaries = []
        
        for result in matched_jobs:
            job = result.get('job', result)
            salary_str = job.get('salary', '')
            if salary_str and salary_str != 'Not specified':
                min_sal, max_sal = SalaryAnalyzer.extract_salary_from_text_regex(salary_str)
                if min_sal and max_sal:
                    salaries.append((min_sal, max_sal))
            
            description = job.get('description', '')
            if description:
                min_sal, max_sal = SalaryAnalyzer.extract_salary_from_text_regex(description[:5000])
                if min_sal and max_sal:
                    salaries.append((min_sal, max_sal))
        
        if not salaries:
            return {'min': 45000, 'max': 55000, 'ideal': 50000}
        
        avg_min = int(sum(s[0] for s in salaries) / len(salaries))
        avg_max = int(sum(s[1] for s in salaries) / len(salaries))
        
        return {
            'min': avg_min,
            'max': avg_max,
            'ideal': (avg_min + avg_max) // 2
        }


# Backward compatibility aliases for function-style imports
def extract_salary_from_text_regex(text: str) -> Tuple[Optional[int], Optional[int]]:
    """Backward compatibility wrapper for SalaryAnalyzer.extract_salary_from_text_regex"""
    return SalaryAnalyzer.extract_salary_from_text_regex(text)


def extract_salary_from_text(text: str, config=None) -> Tuple[Optional[int], Optional[int]]:
    """Backward compatibility wrapper for SalaryAnalyzer.extract_salary_from_text"""
    return SalaryAnalyzer.extract_salary_from_text(text, config)


def filter_jobs_by_salary(
    jobs: List[Dict],
    min_salary: Optional[float] = None,
    expected_salary: Optional[float] = None,
    tolerance: float = 0.2
) -> List[Dict]:
    """Backward compatibility wrapper for SalaryAnalyzer.filter_jobs_by_salary"""
    return SalaryAnalyzer.filter_jobs_by_salary(jobs, min_salary, expected_salary, tolerance)


def calculate_salary_band(
    matched_jobs: List[Dict] = None,
    expected: float = None
) -> Dict[str, float]:
    """Backward compatibility wrapper for SalaryAnalyzer.calculate_salary_band"""
    return SalaryAnalyzer.calculate_salary_band(matched_jobs, expected)
