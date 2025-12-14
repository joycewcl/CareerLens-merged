#!/usr/bin/env python3
"""
Azure OpenAI 404 Error Diagnostic Script
=========================================
This script diagnoses common causes of "Resource not found" (404) errors
when connecting to Azure OpenAI.

Run with: python3 tests/diagnose_azure_404.py
"""

import sys
import os
from pathlib import Path

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


def print_header(text):
    print(f"\n{'='*60}")
    print(f" {text}")
    print(f"{'='*60}")


def print_check(name, passed, details=""):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} - {name}")
    if details:
        print(f"        {details}")


def print_warning(text):
    print(f"⚠️  {text}")


def print_info(text):
    print(f"ℹ️  {text}")


def diagnose_azure_openai():
    """Run comprehensive Azure OpenAI configuration diagnostics."""
    
    print_header("🔍 Azure OpenAI 404 Error Diagnostic")
    
    issues_found = []
    
    # =========================================================================
    # 1. Check if secrets are configured
    # =========================================================================
    print_header("1️⃣ Checking API Key and Endpoint Configuration")
    
    # Try to load from config
    try:
        from config import Config
        Config._initialized = False
        Config.setup()
        
        api_key = Config.AZURE_OPENAI_API_KEY
        endpoint = Config.AZURE_OPENAI_ENDPOINT
        api_version = Config.AZURE_OPENAI_API_VERSION
        deployment = Config.AZURE_OPENAI_DEPLOYMENT
        embedding_deployment = Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        
        print_check("Config module loads", True)
        
    except Exception as e:
        print_check("Config module loads", False, str(e))
        issues_found.append("Config module failed to load")
        return issues_found
    
    # Check API key
    if api_key:
        masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
        print_check("AZURE_OPENAI_API_KEY", True, f"Key present: {masked_key}")
    else:
        print_check("AZURE_OPENAI_API_KEY", False, "Not configured!")
        issues_found.append("Missing AZURE_OPENAI_API_KEY")
    
    # Check endpoint
    if endpoint:
        print_check("AZURE_OPENAI_ENDPOINT", True, f"Value: {endpoint}")
        
        # Endpoint format checks
        print("\n  Endpoint Format Analysis:")
        
        # Check for trailing slash
        if endpoint.endswith('/'):
            print_warning("  Endpoint ends with '/' - this is handled by the code")
        
        # Check for /openai suffix
        if endpoint.endswith('/openai') or endpoint.endswith('/openai/'):
            print_info("  Endpoint contains '/openai' suffix - code strips this automatically")
        
        # Check for correct Azure format
        if '.openai.azure.com' in endpoint:
            print_check("  Azure OpenAI format", True, "Standard Azure format detected")
        elif '.azure-api.net' in endpoint:
            print_check("  Azure API Management format", True, "APIM format detected")
        elif 'api.openai.com' in endpoint:
            print_check("  OpenAI format", False, "This is OpenAI, not Azure OpenAI!")
            issues_found.append("Using OpenAI endpoint instead of Azure OpenAI")
        else:
            print_warning("  Non-standard endpoint format - verify with Azure Portal")
            
    else:
        print_check("AZURE_OPENAI_ENDPOINT", False, "Not configured!")
        issues_found.append("Missing AZURE_OPENAI_ENDPOINT")
    
    # Check API version
    print_check("AZURE_OPENAI_API_VERSION", True if api_version else False, 
                f"Value: {api_version}" if api_version else "Using default")
    
    # Check deployments
    print(f"\n  Deployment Configuration:")
    print(f"    Chat/Completion: {deployment or 'gpt-4o-mini (default)'}")
    print(f"    Embeddings: {embedding_deployment or 'text-embedding-3-small (default)'}")
    
    if not deployment:
        print_warning("  No custom AZURE_OPENAI_DEPLOYMENT set - using 'gpt-4o-mini'")
    if not embedding_deployment:
        print_warning("  No custom AZURE_OPENAI_EMBEDDING_DEPLOYMENT set - using 'text-embedding-3-small'")
    
    # =========================================================================
    # 2. Check URL construction
    # =========================================================================
    print_header("2️⃣ Checking URL Construction")
    
    if endpoint:
        # Simulate URL construction from api_clients.py
        test_endpoint = endpoint.rstrip('/')
        if test_endpoint.endswith('/openai'):
            test_endpoint = test_endpoint[:-7]
        
        test_deployment = deployment or "gpt-4o-mini"
        test_api_version = api_version or "2024-02-01"
        
        chat_url = f"{test_endpoint}/openai/deployments/{test_deployment}/chat/completions?api-version={test_api_version}"
        embedding_url = f"{test_endpoint}/openai/deployments/{embedding_deployment or 'text-embedding-3-small'}/embeddings?api-version={test_api_version}"
        
        print(f"  Constructed Chat URL:")
        print(f"    {chat_url}")
        print(f"\n  Constructed Embedding URL:")
        print(f"    {embedding_url}")
        
        # Common URL issues
        if '//openai' in chat_url:
            print_warning("  Double slash detected in URL - may cause issues")
            issues_found.append("Double slash in constructed URL")
        
        if 'deployments/None/' in chat_url:
            print_check("Deployment in URL", False, "Deployment is None!")
            issues_found.append("Deployment name is None")
    
    # =========================================================================
    # 3. Test actual connectivity
    # =========================================================================
    print_header("3️⃣ Testing API Connectivity")
    
    if api_key and endpoint:
        import requests
        
        # Build test URL
        test_endpoint = endpoint.rstrip('/')
        if test_endpoint.endswith('/openai'):
            test_endpoint = test_endpoint[:-7]
        
        test_deployment = deployment or "gpt-4o-mini"
        test_url = f"{test_endpoint}/openai/deployments/{test_deployment}/chat/completions?api-version={api_version or '2024-02-01'}"
        
        headers = {
            "api-key": api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": [{"role": "user", "content": "Say 'test' only"}],
            "max_tokens": 5
        }
        
        print(f"  Testing URL: {test_url[:80]}...")
        
        try:
            response = requests.post(test_url, headers=headers, json=payload, timeout=30)
            
            print(f"\n  Response Status: {response.status_code}")
            
            if response.status_code == 200:
                print_check("API Connection", True, "Successfully connected!")
                try:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    print(f"    Response: '{content}'")
                except:
                    pass
                    
            elif response.status_code == 404:
                print_check("API Connection", False, "404 Not Found!")
                print(f"\n  ❌ 404 ERROR DIAGNOSIS:")
                
                try:
                    error_body = response.json()
                    error_msg = error_body.get('error', {}).get('message', str(error_body))
                    print(f"    Error message: {error_msg}")
                    
                    if 'deployment' in error_msg.lower():
                        issues_found.append(f"Deployment '{test_deployment}' not found in Azure")
                        print(f"\n  🔧 FIX: The deployment name '{test_deployment}' doesn't exist.")
                        print(f"     Check Azure Portal → Your OpenAI Resource → Deployments")
                        print(f"     Use the exact deployment name from there.")
                    elif 'resource' in error_msg.lower():
                        issues_found.append("Azure resource not found")
                        print(f"\n  🔧 FIX: The Azure OpenAI resource doesn't exist or endpoint is wrong.")
                        print(f"     Check your AZURE_OPENAI_ENDPOINT matches Azure Portal exactly.")
                except:
                    print(f"    Raw response: {response.text[:500]}")
                    issues_found.append("404 Resource Not Found")
                    
            elif response.status_code == 401:
                print_check("API Connection", False, "401 Unauthorized!")
                print(f"  🔧 FIX: API key is invalid or expired.")
                issues_found.append("Invalid API key")
                
            elif response.status_code == 403:
                print_check("API Connection", False, "403 Forbidden!")
                print(f"  🔧 FIX: API key doesn't have access to this resource.")
                issues_found.append("API key lacks permissions")
                
            elif response.status_code == 429:
                print_check("API Connection", True, "Rate limited but connection works")
                print(f"  ⚠️ Rate limit hit - but this confirms connection works!")
                
            else:
                print_check("API Connection", False, f"Unexpected status: {response.status_code}")
                print(f"    Response: {response.text[:300]}")
                issues_found.append(f"Unexpected HTTP status: {response.status_code}")
                
        except requests.exceptions.Timeout:
            print_check("API Connection", False, "Request timed out")
            issues_found.append("Connection timeout")
            
        except requests.exceptions.ConnectionError as e:
            print_check("API Connection", False, f"Connection error: {e}")
            issues_found.append("Cannot connect to endpoint")
            
        except Exception as e:
            print_check("API Connection", False, f"Error: {e}")
            issues_found.append(str(e))
    else:
        print_warning("  Skipping connectivity test - credentials not configured")
    
    # =========================================================================
    # 4. Check code paths
    # =========================================================================
    print_header("4️⃣ Checking Code Configuration Paths")
    
    print("  Files that construct Azure OpenAI URLs:")
    print("    • config.py - Central configuration")
    print("    • services/azure_openai.py - Uses Config directly with openai SDK")
    print("    • utils/api_clients.py - Uses REST API directly (strips /openai suffix)")
    print("    • core/resume_parser.py - Uses Config.AZURE_ENDPOINT alias")
    
    # Check if there are deployment mismatches
    if deployment:
        print(f"\n  Deployment from config: {deployment}")
    
    # Check hardcoded values in api_clients
    print(f"\n  Hardcoded in utils/api_clients.py:")
    print(f"    • Chat deployment: 'gpt-4o-mini'")
    print(f"    • Embedding deployment: 'text-embedding-3-small'")
    print(f"    • API version: '2024-02-01'")
    
    if deployment and deployment != 'gpt-4o-mini':
        print_warning(f"  Mismatch! Config has '{deployment}' but api_clients.py uses 'gpt-4o-mini'")
        issues_found.append("Deployment name mismatch between config and api_clients.py")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print_header("📋 DIAGNOSTIC SUMMARY")
    
    if not issues_found:
        print("✅ No issues detected! If you're still getting 404 errors:")
        print("   1. Verify deployment name in Azure Portal")
        print("   2. Ensure the deployment is fully provisioned (not still deploying)")
        print("   3. Check if the model is available in your Azure region")
    else:
        print(f"❌ Found {len(issues_found)} issue(s):\n")
        for i, issue in enumerate(issues_found, 1):
            print(f"   {i}. {issue}")
        
        print("\n🔧 RECOMMENDED FIXES:")
        print("   1. Go to Azure Portal → Your OpenAI Resource → Deployments")
        print("   2. Copy the exact deployment name (e.g., 'gpt-4-deployment')")
        print("   3. Update .streamlit/secrets.toml:")
        print("      AZURE_OPENAI_DEPLOYMENT = 'your-exact-deployment-name'")
        print("   4. Ensure endpoint format is correct:")
        print("      AZURE_OPENAI_ENDPOINT = 'https://YOUR-RESOURCE.openai.azure.com'")
    
    return issues_found


if __name__ == "__main__":
    issues = diagnose_azure_openai()
    sys.exit(1 if issues else 0)
