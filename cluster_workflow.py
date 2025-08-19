#!/usr/bin/env python3
"""
Cluster transfer and testing script for julia2 HPC
Automates the process of transferring code and running tests on the cluster
"""
import subprocess
import sys
import os
from pathlib import Path

# Cluster configuration
CLUSTER_USER = "s391129"
CLUSTER_HOST = "julia2.hpc.uni-wuerzburg.de"
CLUSTER_PATH = "/home/s391129/negotiation_platform"
LOCAL_PATH = "./negotiation_platform"

def run_command(cmd, description=""):
    """Run a command and return success status"""
    if description:
        print(f"🔧 {description}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Success")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Failed")
            if result.stderr.strip():
                print(f"   Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def test_local_code():
    """Run local tests before transfer"""
    print("\n🧪 RUNNING LOCAL TESTS")
    print("=" * 50)
    
    if not run_command("python test_all_modules.py", "Testing all modules locally"):
        print("❌ Local tests failed! Fix issues before cluster transfer.")
        return False
    
    print("✅ All local tests passed!")
    return True

def transfer_to_cluster():
    """Transfer code to cluster using scp (Windows compatible)"""
    print("\n📤 TRANSFERRING CODE TO CLUSTER")
    print("=" * 50)
    
    # Create remote directory
    create_dir_cmd = f'ssh {CLUSTER_USER}@{CLUSTER_HOST} "mkdir -p {CLUSTER_PATH}"'
    if not run_command(create_dir_cmd, "Creating remote directory"):
        return False
    
    # Transfer code using scp with recursive flag (Windows compatible)
    scp_cmd = f'scp -r {LOCAL_PATH} {CLUSTER_USER}@{CLUSTER_HOST}:/home/{CLUSTER_USER}/'
    if not run_command(scp_cmd, "Copying code to cluster"):
        return False
    
    # Also transfer test scripts
    test_files = ["test_all_modules.py", "test_runner.py"]
    for test_file in test_files:
        if Path(test_file).exists():
            transfer_cmd = f'scp {test_file} {CLUSTER_USER}@{CLUSTER_HOST}:/home/{CLUSTER_USER}/'
            run_command(transfer_cmd, f"Transferring {test_file}")
    
    print("✅ Code transfer completed!")
    return True

def run_cluster_tests():
    """Run tests on the cluster"""
    print("\n🧪 RUNNING TESTS ON CLUSTER")
    print("=" * 50)
    
    # Test if Python is available
    python_test_cmd = f'ssh {CLUSTER_USER}@{CLUSTER_HOST} "python3 --version"'
    if not run_command(python_test_cmd, "Checking Python on cluster"):
        print("⚠️  Python3 not found, trying python...")
        python_test_cmd = f'ssh {CLUSTER_USER}@{CLUSTER_HOST} "python --version"'
        if not run_command(python_test_cmd, "Checking Python on cluster"):
            print("❌ Python not available on cluster!")
            return False
    
    # Run the comprehensive test on cluster
    cluster_test_cmd = f'ssh {CLUSTER_USER}@{CLUSTER_HOST} "cd /home/{CLUSTER_USER} && python3 test_all_modules.py"'
    if not run_command(cluster_test_cmd, "Running comprehensive tests on cluster"):
        # Try with python instead of python3
        cluster_test_cmd = f'ssh {CLUSTER_USER}@{CLUSTER_HOST} "cd /home/{CLUSTER_USER} && python test_all_modules.py"'
        if not run_command(cluster_test_cmd, "Running tests with python"):
            return False
    
    print("✅ Cluster tests completed successfully!")
    return True

def check_cluster_environment():
    """Check the cluster environment and installed packages"""
    print("\n🔍 CHECKING CLUSTER ENVIRONMENT")
    print("=" * 50)
    
    # Check Python packages
    packages_to_check = ["torch", "transformers", "numpy", "pyyaml"]
    
    for package in packages_to_check:
        check_cmd = f'ssh {CLUSTER_USER}@{CLUSTER_HOST} "python3 -c \\"import {package}; print(\\\'✅ {package} available\\\')\\" 2>/dev/null || echo \\"❌ {package} not found\\""'
        run_command(check_cmd, f"Checking {package}")

def interactive_cluster_session():
    """Start an interactive SSH session to the cluster"""
    print("\n🖥️  STARTING INTERACTIVE CLUSTER SESSION")
    print("=" * 50)
    print("You will be connected to the cluster. Type 'exit' to return.")
    print("Your code is available at: /home/s391129/negotiation_platform")
    print("")
    
    ssh_cmd = f'ssh {CLUSTER_USER}@{CLUSTER_HOST}'
    os.system(ssh_cmd)

def main():
    """Main function with menu"""
    print("🚀 JULIA2 CLUSTER WORKFLOW")
    print("=" * 50)
    
    while True:
        print("\nSelect an option:")
        print("1. 🧪 Test code locally")
        print("2. 📤 Transfer code to cluster")
        print("3. 🧪 Run tests on cluster")
        print("4. 🔍 Check cluster environment")
        print("5. 🚀 Full workflow (test + transfer + cluster test)")
        print("6. 🖥️  Connect to cluster (interactive)")
        print("7. ❌ Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == "1":
            test_local_code()
        
        elif choice == "2":
            transfer_to_cluster()
        
        elif choice == "3":
            run_cluster_tests()
        
        elif choice == "4":
            check_cluster_environment()
        
        elif choice == "5":
            print("\n🚀 FULL WORKFLOW")
            print("=" * 50)
            if test_local_code():
                if transfer_to_cluster():
                    run_cluster_tests()
                else:
                    print("❌ Transfer failed!")
            else:
                print("❌ Local tests failed!")
        
        elif choice == "6":
            interactive_cluster_session()
        
        elif choice == "7":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-7.")

if __name__ == "__main__":
    main()
