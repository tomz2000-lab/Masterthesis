#!/usr/bin/env python3
import subprocess
import sys
import os

# Read the enhanced session_manager.py content
with open('negotiation_platform/core/session_manager.py', 'r') as f:
    content = f.read()

# Create a temporary script to upload the content
script_content = f'''#!/bin/bash
cat > ~/Masterthesis/negotiation_platform/core/session_manager.py << 'EOF'
{content}
EOF
echo "✅ session_manager.py updated successfully!"
'''

# Write the script to a temporary file
with open('upload_script.sh', 'w') as f:
    f.write(script_content)

print("📁 Created upload script: upload_script.sh")
print("📤 Now upload this script to the cluster and run it")
