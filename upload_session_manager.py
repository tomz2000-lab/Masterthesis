#!/usr/bin/env python3
"""
Quick script to upload session_manager.py to the cluster
"""
import subprocess
import sys
import os

def upload_file():
    """Upload session_manager.py to cluster"""
    source_file = "negotiation_platform/core/session_manager.py"
    dest_path = "julia2.hpc.uni-wuerzburg.de:~/Masterthesis/negotiation_platform/core/"
    
    if not os.path.exists(source_file):
        print(f"❌ Source file not found: {source_file}")
        return False
    
    print(f"🔄 Uploading {source_file} to cluster...")
    try:
        result = subprocess.run([
            "scp", source_file, dest_path
        ], check=True, capture_output=True, text=True)
        print("✅ Upload successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Upload failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

if __name__ == "__main__":
    success = upload_file()
    sys.exit(0 if success else 1)
