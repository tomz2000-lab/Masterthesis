#!/bin/bash
# Real-time model output monitor for negotiation platform

echo "🔍 REAL-TIME MODEL OUTPUT MONITOR"
echo "=================================="

# Function to get latest job output
get_latest_output() {
    LATEST_FILE=$(ls -t negotiation_*.out 2>/dev/null | head -1)
    if [ -z "$LATEST_FILE" ]; then
        echo "❌ No output files found"
        return 1
    fi
    echo "📄 Monitoring: $LATEST_FILE"
    echo
}

# Function to show live model analysis
show_live_analysis() {
    LATEST_FILE=$(ls -t negotiation_*.out 2>/dev/null | head -1)
    if [ -n "$LATEST_FILE" ]; then
        echo "=== LIVE MODEL PERFORMANCE ==="
        echo "Model A (Llama) successful responses: $(grep -c 'Successfully parsed JSON' $LATEST_FILE)"
        echo "Model B (DialoGPT) failed responses: $(grep -c 'Could not parse action' $LATEST_FILE)"
        echo "Current agreement status: $(grep 'Agreement reached:' $LATEST_FILE | tail -1 | cut -d: -f2)"
        echo
        
        echo "=== LATEST OFFERS ==="
        grep 'price.*42000' $LATEST_FILE | tail -3
        echo
    fi
}

# Function to monitor job status
monitor_jobs() {
    echo "=== CURRENT JOBS ==="
    squeue -u s391129 2>/dev/null || echo "No active jobs"
    echo
}

# Main monitoring loop
while true; do
    clear
    echo "🔍 NEGOTIATION PLATFORM MONITOR - $(date)"
    echo "==========================================="
    echo
    
    monitor_jobs
    get_latest_output
    show_live_analysis
    
    echo "Press Ctrl+C to exit, or wait 30 seconds for refresh..."
    sleep 30
done
