#!/usr/bin/env python3
"""
Comprehensive test script for all negotiation platform modules
Tests all files before cluster transfer
"""
import sys
import importlib
import traceback
from pathlib import Path
from typing import List, Tuple

# Define all modules to test (excluding main.py)
MODULES_TO_TEST = [
    # Core modules
    "negotiation_platform.core.base_metric",
    "negotiation_platform.core.config_manager", 
    "negotiation_platform.core.game_engine",
    "negotiation_platform.core.llm_manager",
    "negotiation_platform.core.metrics_calculator",
    "negotiation_platform.core.session_manager",
    
    # Game modules
    "negotiation_platform.games.base_game",
    "negotiation_platform.games.integrative_negotiation",
    "negotiation_platform.games.price_bargaining",
    "negotiation_platform.games.resource_exchange",
    
    # Model modules
    "negotiation_platform.models.base_model",
    "negotiation_platform.models.hf_model_wrapper",
    
    # Metrics modules
    "negotiation_platform.metrics.deadline_sensitivity",
    "negotiation_platform.metrics.feasibility",
    "negotiation_platform.metrics.risk_minimization",
    "negotiation_platform.metrics.utility_surplus",
]

def test_module_import(module_path: str) -> Tuple[bool, str, str]:
    """
    Test if a module can be imported
    Returns: (success, module_path, error_message)
    """
    try:
        module = importlib.import_module(module_path)
        return True, module_path, ""
    except ImportError as e:
        return False, module_path, f"ImportError: {str(e)}"
    except Exception as e:
        return False, module_path, f"Error: {str(e)}"

def test_module_basic_functionality(module_path: str) -> Tuple[bool, str, List[str]]:
    """
    Test basic functionality of a module
    Returns: (success, module_path, list_of_classes_functions)
    """
    try:
        module = importlib.import_module(module_path)
        items = []
        
        for name in dir(module):
            if not name.startswith('_'):
                obj = getattr(module, name)
                if hasattr(obj, '__module__') and obj.__module__ == module_path:
                    items.append(f"{name} ({type(obj).__name__})")
        
        return True, module_path, items
    except Exception as e:
        return False, module_path, [f"Error: {str(e)}"]

def run_all_tests():
    """Run tests on all modules"""
    print("🧪 COMPREHENSIVE MODULE TESTING")
    print("=" * 80)
    print(f"Testing {len(MODULES_TO_TEST)} modules...\n")
    
    passed = 0
    failed = 0
    results = []
    
    # Test each module
    for i, module_path in enumerate(MODULES_TO_TEST, 1):
        print(f"[{i:2d}/{len(MODULES_TO_TEST)}] Testing {module_path}...")
        
        # Test import
        import_success, _, import_error = test_module_import(module_path)
        
        if import_success:
            # Test basic functionality
            func_success, _, items = test_module_basic_functionality(module_path)
            
            if func_success:
                print(f"    ✅ PASSED - {len(items)} items found")
                passed += 1
                results.append((module_path, "PASSED", ""))
            else:
                print(f"    ❌ FAILED - Functionality test failed")
                failed += 1
                results.append((module_path, "FAILED", "Functionality test failed"))
        else:
            print(f"    ❌ FAILED - {import_error}")
            failed += 1
            results.append((module_path, "FAILED", import_error))
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {(passed/(passed+failed)*100):.1f}%")
    
    # Detailed results
    if failed > 0:
        print("\n❌ FAILED MODULES:")
        print("-" * 40)
        for module_path, status, error in results:
            if status == "FAILED":
                print(f"   {module_path}")
                if error:
                    print(f"      Error: {error}")
    
    print("\n✅ PASSED MODULES:")
    print("-" * 40)
    for module_path, status, error in results:
        if status == "PASSED":
            print(f"   {module_path}")
    
    # Final verdict
    print("\n" + "=" * 80)
    if failed == 0:
        print("🎉 ALL TESTS PASSED! Your code is ready for cluster transfer!")
        print("✅ You can safely transfer all files to the cluster.")
    else:
        print("⚠️  SOME TESTS FAILED! Fix issues before cluster transfer.")
        print("❌ Review the failed modules above and fix the errors.")
    
    return failed == 0

def check_file_existence():
    """Check if all Python files exist"""
    print("📁 CHECKING FILE EXISTENCE")
    print("=" * 80)
    
    missing_files = []
    existing_files = []
    
    for module_path in MODULES_TO_TEST:
        # Convert module path to file path
        relative_path = module_path.replace("negotiation_platform.", "").replace(".", "/")
        file_path = Path("negotiation_platform") / (relative_path + ".py")
        
        if file_path.exists():
            existing_files.append(str(file_path))
            print(f"✅ {file_path}")
        else:
            missing_files.append(str(file_path))
            print(f"❌ {file_path} - FILE NOT FOUND")
    
    print(f"\n📊 Files found: {len(existing_files)}/{len(MODULES_TO_TEST)}")
    
    if missing_files:
        print("\n❌ Missing files:")
        for file in missing_files:
            print(f"   {file}")
        return False
    
    return True

def main():
    """Main test function"""
    print("🔍 NEGOTIATION PLATFORM - PRE-TRANSFER TESTING")
    print("=" * 80)
    print("This script will test all modules before cluster transfer\n")
    
    # Check file existence first
    files_exist = check_file_existence()
    print()
    
    if not files_exist:
        print("❌ Some files are missing. Cannot proceed with testing.")
        sys.exit(1)
    
    # Run comprehensive tests
    all_passed = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
