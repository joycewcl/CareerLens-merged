#!/usr/bin/env python3
"""
Test HKUST Azure API Management Endpoint
=========================================
Diagnoses connectivity issues with the HKUST APIM gateway.
"""

import requests
import sys

# HKUST endpoint configuration
ENDPOINT = "https://hkust.azure-api.net/openai"
# Note: API key should be passed as argument or environment variable for security

def test_endpoint(api_key):
    """Test various deployment configurations against the HKUST endpoint."""
    
    print("=" * 60)
    print(" HKUST Azure OpenAI Endpoint Diagnostic")
    print("=" * 60)
    
    # The code strips /openai suffix, so test both
    base_endpoints = [
        ENDPOINT,  # Original: https://hkust.azure-api.net/openai
        ENDPOINT.replace('/openai', ''),  # Stripped: https://hkust.azure-api.net
    ]
    
    # Common deployment names to try
    deployments_to_try = [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4",
        "gpt-35-turbo",
        "gpt-3.5-turbo",
        "chatgpt",
        "text-davinci-003",
    ]
    
    # API versions to try
    api_versions = [
        "2024-02-15-preview",
        "2024-02-01",
        "2023-12-01-preview",
        "2023-05-15",
    ]
    
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json"
    }
    
    payload = {
        "messages": [{"role": "user", "content": "Say 'test'"}],
        "max_tokens": 5
    }
    
    print(f"\n📍 Testing endpoint: {ENDPOINT}")
    print(f"🔑 API Key: {api_key[:8]}...{api_key[-4:]}")
    
    # First, try the exact URL pattern the app uses
    print("\n" + "-" * 60)
    print("1️⃣ Testing with app's URL construction pattern")
    print("-" * 60)
    
    # Simulate what utils/api_clients.py does
    test_endpoint = ENDPOINT.rstrip('/')
    if test_endpoint.endswith('/openai'):
        test_endpoint = test_endpoint[:-7]  # Remove /openai
    
    print(f"   Base (after stripping /openai): {test_endpoint}")
    
    for deployment in deployments_to_try[:4]:  # Try top 4
        for api_version in api_versions[:2]:  # Try top 2 versions
            url = f"{test_endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
            
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=10)
                status = response.status_code
                
                if status == 200:
                    print(f"\n   ✅ SUCCESS! Deployment: {deployment}, API Version: {api_version}")
                    print(f"      URL: {url}")
                    try:
                        result = response.json()
                        content = result['choices'][0]['message']['content']
                        print(f"      Response: '{content}'")
                    except:
                        pass
                    return deployment, api_version
                    
                elif status == 404:
                    print(f"   ❌ 404 - {deployment} (v{api_version})")
                    
                elif status == 401:
                    print(f"   ⚠️ 401 Unauthorized - API key invalid")
                    return None, None
                    
                elif status == 403:
                    print(f"   ⚠️ 403 Forbidden - No access to {deployment}")
                    
                else:
                    print(f"   ⚠️ {status} - {deployment}")
                    
            except requests.exceptions.Timeout:
                print(f"   ⏱️ Timeout - {deployment}")
            except Exception as e:
                print(f"   ❌ Error - {deployment}: {e}")
    
    # Try without the /openai in the path (some APIM configs)
    print("\n" + "-" * 60)
    print("2️⃣ Testing APIM alternate patterns")
    print("-" * 60)
    
    alternate_patterns = [
        # Pattern 1: /openai already in base, so don't add again
        f"{ENDPOINT}/deployments/{{deployment}}/chat/completions?api-version={{api_version}}",
        # Pattern 2: Standard with base having /openai
        f"{ENDPOINT.rstrip('/')}/deployments/{{deployment}}/chat/completions?api-version={{api_version}}",
    ]
    
    for pattern in alternate_patterns:
        for deployment in deployments_to_try[:3]:
            for api_version in api_versions[:1]:
                url = pattern.format(deployment=deployment, api_version=api_version)
                
                try:
                    response = requests.post(url, headers=headers, json=payload, timeout=10)
                    status = response.status_code
                    
                    if status == 200:
                        print(f"\n   ✅ SUCCESS with alternate pattern!")
                        print(f"      URL: {url}")
                        return deployment, api_version
                    elif status == 404:
                        print(f"   ❌ 404 - {url[:70]}...")
                    else:
                        print(f"   ⚠️ {status} - {url[:70]}...")
                        
                except Exception as e:
                    print(f"   ❌ {url[:50]}... - {e}")
    
    # Try to list models/deployments (if endpoint supports it)
    print("\n" + "-" * 60)
    print("3️⃣ Attempting to discover available deployments")
    print("-" * 60)
    
    discovery_urls = [
        f"{test_endpoint}/openai/deployments?api-version=2024-02-01",
        f"{test_endpoint}/openai/models?api-version=2024-02-01",
        f"{ENDPOINT}/deployments?api-version=2024-02-01",
        f"{ENDPOINT}/models?api-version=2024-02-01",
    ]
    
    for url in discovery_urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                print(f"   ✅ Found deployments endpoint: {url}")
                try:
                    data = response.json()
                    print(f"      Response: {str(data)[:500]}")
                except:
                    print(f"      Response: {response.text[:500]}")
                break
            else:
                print(f"   {response.status_code} - {url}")
        except Exception as e:
            print(f"   ❌ {url[:50]}... - {e}")
    
    return None, None


def print_recommendations(working_deployment, working_version):
    """Print configuration recommendations."""
    
    print("\n" + "=" * 60)
    print(" 📋 RECOMMENDATIONS")
    print("=" * 60)
    
    if working_deployment and working_version:
        print(f"""
✅ Working configuration found!

Update your .streamlit/secrets.toml:

```toml
AZURE_OPENAI_ENDPOINT = "https://hkust.azure-api.net/openai"
AZURE_OPENAI_API_KEY = "your-api-key"
AZURE_OPENAI_DEPLOYMENT = "{working_deployment}"
AZURE_OPENAI_API_VERSION = "{working_version}"
```
""")
    else:
        print("""
❌ Could not find a working deployment.

This could mean:
1. The deployment names tested don't match what's available
2. The API key doesn't have access to any deployments
3. The APIM gateway has a different URL pattern

📞 Contact HKUST IT to get:
   - The exact deployment name(s) available
   - The correct URL pattern for the APIM gateway
   - Confirmation that your API key has proper permissions

Common HKUST deployment names might be:
   - "gpt-4" or "gpt4"
   - "gpt-35-turbo" or "gpt35turbo"
   - A custom name like "hkust-gpt4"
""")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_hkust_endpoint.py YOUR_API_KEY")
        print("\nThis will test various deployment names against the HKUST endpoint.")
        sys.exit(1)
    
    api_key = sys.argv[1]
    deployment, version = test_endpoint(api_key)
    print_recommendations(deployment, version)
