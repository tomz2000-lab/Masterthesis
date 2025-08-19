#!/usr/bin/env python3
"""
Simple Windows-compatible transfer script for julia2 cluster
Uses basic scp commands that work on Windows
"""
import subprocess
import sys
import os
from pathlib import Path

def test_ssh_connection():
    """Test if SSH connection works"""
    print("🔍 Testing SSH connection...")
    cmd = "ssh -o ConnectTimeout=10 s391129@julia2.hpc.uni-wuerzburg.de echo 'Connection successful'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ SSH connection working")
        return True
    else:
        print("❌ SSH connection failed")
        print(f"Error: {result.stderr}")
        return False

def run_local_tests():
    """Run local tests"""
    print("🧪 Running local tests...")
    result = subprocess.run("python test_all_modules.py", shell=True)
    return result.returncode == 0

def transfer_files():
    """Transfer files using simple scp commands"""
    print("\n📤 Transferring files to cluster...")
    
    # Files and directories to transfer
    items_to_transfer = [
        ("negotiation_platform", "directory"),
        ("test_all_modules.py", "file"),
        ("test_runner.py", "file"),
    ]
    
    for item, item_type in items_to_transfer:
        if not Path(item).exists():
            print(f"⚠️  {item} not found, skipping...")
            continue
            
        print(f"📁 Transferring {item}...")
        
        if item_type == "directory":
            # For directories, use scp -r
            cmd = f'scp -r "{item}" s391129@julia2.hpc.uni-wuerzburg.de:/home/s391129/'
        else:
            # For files, use simple scp
            cmd = f'scp "{item}" s391129@julia2.hpc.uni-wuerzburg.de:/home/s391129/'
        
        result = subprocess.run(cmd, shell=True)
        
        if result.returncode == 0:
            print(f"✅ {item} transferred successfully")
        else:
            print(f"❌ Failed to transfer {item}")
            return False
    
    return True

def main():
    print("🚀 SIMPLE CLUSTER TRANSFER")
    print("=" * 40)
    
    # Test SSH connection first
    if not test_ssh_connection():
        print("\n❌ Cannot connect to cluster. Check your SSH setup.")
        print("Make sure you can run: ssh s391129@julia2.hpc.uni-wuerzburg.de")
        return
    
    # Ask user if they want to test first
    test_choice = input("\nRun local tests first? (y/n): ").lower().strip()
    
    if test_choice in ['y', 'yes']:
        if not run_local_tests():
            print("❌ Local tests failed!")
            choice = input("Continue with transfer anyway? (y/n): ").lower().strip()
            if choice not in ['y', 'yes']:
                print("Transfer cancelled.")
                return
        else:
            print("✅ Local tests passed!")
    
    # Transfer files
    if transfer_files():
        print("\n✅ Transfer completed successfully!")
        print("\n🧪 Next steps:")
        print("1. Connect to cluster: ssh s391129@julia2.hpc.uni-wuerzburg.de")
        print("2. Test on cluster: python3 test_all_modules.py")
        print("3. Run your code: python3 negotiation_platform/main.py")
    else:
        print("\n❌ Transfer failed!")

if __name__ == "__main__":
    main()
