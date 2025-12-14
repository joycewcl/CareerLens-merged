#!/usr/bin/env python3
"""
CareerLens Setup Verification Script
=====================================
This script verifies that all required dependencies are installed
and the application environment is properly configured.

Run this script after installing requirements.txt to ensure
everything is set up correctly.

Usage:
    python verify_setup.py
"""

import sys
import os
import importlib
from typing import Tuple


def check_python_version() -> bool:
    """Check if Python version is 3.8 or higher."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if (version.major, version.minor) >= (3, 8):
        print("✅ Python version is compatible (3.8+)")
        return True
    else:
        print(f"❌ Python version {version.major}.{version.minor} is not supported. Please use Python 3.8 or higher.")
        return False


def check_module(module_name: str, import_path: str = None) -> bool:
    """Check if a Python module can be imported."""
    try:
        if import_path:
            module = importlib.import_module(import_path)
        else:
            module = importlib.import_module(module_name)
        
        # Try to get version if available
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {module_name} (version: {version})")
        return True
    except ImportError as e:
        print(f"❌ {module_name} - NOT INSTALLED")
        print(f"   Error: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {module_name} - WARNING: {e}")
        return True  # Module exists but has some issue


def check_required_modules() -> Tuple[int, int]:
    """Check all required modules for the application."""
    print("\nChecking required modules...")
    print("-" * 50)
    
    # Core modules that must be installed
    required_modules = [
        ("streamlit", "streamlit"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("plotly", "plotly"),
        ("scikit-learn", "sklearn"),
        ("PyPDF2", "PyPDF2"),
        ("python-docx", "docx"),
        ("pinecone-client", "pinecone"),
        ("sentence-transformers", "sentence_transformers"),
        ("openai", "openai"),
        ("tiktoken", "tiktoken"),
        ("reportlab", "reportlab"),
        ("requests", "requests"),
    ]
    
    passed = 0
    failed = 0
    
    for display_name, import_name in required_modules:
        if check_module(display_name, import_name):
            passed += 1
        else:
            failed += 1
    
    return passed, failed


def check_app_files() -> bool:
    """Check if main application files exist."""
    print("\nChecking application files...")
    print("-" * 50)
    
    required_files = [
        "streamlit_app.py",
        "config.py",
        "requirements.txt",
    ]
    
    all_exist = True
    for filename in required_files:
        if os.path.exists(filename):
            print(f"✅ {filename}")
        else:
            print(f"❌ {filename} - NOT FOUND")
            all_exist = False
    
    # Check modules directory
    if os.path.exists("modules") and os.path.isdir("modules"):
        print(f"✅ modules/ directory")
    else:
        print(f"⚠️  modules/ directory - NOT FOUND (required for Market Dashboard features)")
        all_exist = False
    
    return all_exist


def check_streamlit_app_imports() -> bool:
    """Test if streamlit_app.py can be compiled without syntax errors."""
    print("\nChecking streamlit_app.py for syntax errors...")
    print("-" * 50)
    
    try:
        with open('streamlit_app.py', 'r') as f:
            code = f.read()
        compile(code, 'streamlit_app.py', 'exec')
        print("✅ streamlit_app.py has no syntax errors")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in streamlit_app.py:")
        print(f"   Line {e.lineno}: {e.msg}")
        return False
    except FileNotFoundError:
        print("❌ streamlit_app.py not found")
        return False
    except Exception as e:
        print(f"⚠️  Warning: {e}")
        return True


def main():
    """Main verification function."""
    print("=" * 50)
    print("CareerLens Setup Verification")
    print("=" * 50)
    
    results = {
        'python_version': False,
        'modules': (0, 0),
        'app_files': False,
        'syntax_check': False,
    }
    
    # Check Python version
    results['python_version'] = check_python_version()
    
    # Check required modules
    results['modules'] = check_required_modules()
    
    # Check application files
    results['app_files'] = check_app_files()
    
    # Check streamlit app syntax
    results['syntax_check'] = check_streamlit_app_imports()
    
    # Summary
    print("\n" + "=" * 50)
    print("Verification Summary")
    print("=" * 50)
    
    passed_modules, failed_modules = results['modules']
    total_modules = passed_modules + failed_modules
    
    print(f"Python version: {'✅ PASS' if results['python_version'] else '❌ FAIL'}")
    print(f"Required modules: {passed_modules}/{total_modules} installed")
    print(f"Application files: {'✅ PASS' if results['app_files'] else '❌ FAIL'}")
    print(f"Syntax check: {'✅ PASS' if results['syntax_check'] else '❌ FAIL'}")
    
    if failed_modules > 0:
        print("\n⚠️  Some modules are missing. Install them with:")
        print("   pip install -r requirements.txt")
    
    # Check if all tests passed
    all_checks_passed = (
        results['python_version'] and 
        failed_modules == 0 and 
        results['app_files'] and 
        results['syntax_check']
    )
    
    if all_checks_passed:
        print("\n✅ All checks passed! Your environment is ready.")
        print("\nYou can now run the application with:")
        print("   streamlit run streamlit_app.py")
        return 0
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
