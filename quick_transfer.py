#!/usr/bin/env python3
"""
Quick transfer script for julia2 cluster
Usage: python quick_transfer.py [--test-first]
"""
import subprocess
import sys
import argparse

def quick_transfer(test_first=True):
    """Quick transfer with optional testing"""
    
    if test_first:
        print("🧪 Running local tests first...")
        result = subprocess.run("python test_all_modules.py", shell=True)
        if result.returncode != 0:
            print("❌ Local tests failed! Transfer cancelled.")
            return False
        print("✅ Local tests passed!")
    
    print("\n📤 Transferring to cluster...")
    
    # Transfer negotiation_platform directory using scp (Windows compatible)
    cmd = "scp -r negotiation_platform s391129@julia2.hpc.uni-wuerzburg.de:/home/s391129/"
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode == 0:
        print("✅ Transfer successful!")
        
        # Also transfer test files
        test_files = ["test_all_modules.py", "test_runner.py"]
        for file in test_files:
            cmd = f"scp {file} s391129@julia2.hpc.uni-wuerzburg.de:/home/s391129/"
            subprocess.run(cmd, shell=True)
        
        print("🧪 Now you can SSH to cluster and run:")
        print("   ssh s391129@julia2.hpc.uni-wuerzburg.de")
        print("   python3 test_all_modules.py")
        return True
    else:
        print("❌ Transfer failed!")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quick transfer to julia2 cluster')
    parser.add_argument('--no-test', action='store_true', help='Skip local testing')
    
    args = parser.parse_args()
    quick_transfer(test_first=not args.no_test)
