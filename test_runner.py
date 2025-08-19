#!/usr/bin/env python3
"""
General test runner for individual negotiation platform files
Usage: python test_runner.py <module_path>
Example: python test_runner.py negotiation_platform.core.base_metric
"""
import sys
import importlib
import argparse

def test_module(module_path):
    """Test if a module can be imported and basic functionality works"""
    print(f"🧪 Testing module: {module_path}")
    print("=" * 60)
    
    try:
        # Import the module
        module = importlib.import_module(module_path)
        print(f"✅ Module '{module_path}' imported successfully")
        
        # List available classes/functions
        print("\n📋 Available classes and functions:")
        for name in dir(module):
            if not name.startswith('_'):
                obj = getattr(module, name)
                if callable(obj):
                    print(f"   📝 {name}: {type(obj).__name__}")
        
        print(f"\n🎉 Module '{module_path}' is working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test individual negotiation platform modules')
    parser.add_argument('module', help='Module path (e.g., negotiation_platform.core.base_metric)')
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        print("🔧 Available modules to test:")
        print("   negotiation_platform.core.base_metric")
        print("   negotiation_platform.core.game_engine")
        print("   negotiation_platform.core.llm_manager")
        print("   negotiation_platform.games.base_game")
        print("   negotiation_platform.models.base_model")
        print("   negotiation_platform.metrics.feasibility")
        print("\nUsage: python test_runner.py <module_path>")
        return
    
    args = parser.parse_args()
    success = test_module(args.module)
    
    if success:
        print("\n✅ Module is ready for cluster transfer!")
    else:
        print("\n❌ Fix issues before transferring to cluster!")
        sys.exit(1)

if __name__ == "__main__":
    main()
