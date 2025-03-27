#!/bin/bash

# Script to run a Python script in the background using nohup with CPU/memory limits

# Default values
CPU_CORES="0-24"
MEMORY_GB=40
SCRIPT_PATH="scenario_tree_mip.py"
LOG_FILE="output_files/output.log"

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu)
            CPU_CORES="$2"
            shift 2
            ;;
        --mem)
            MEMORY_GB="$2"
            shift 2
            ;;
        --) # end of options
            shift
            break
            ;;
        *) # pass remaining args to the Python script
            break
            ;;
    esac
done

# Convert GB to KB
MEMORY_LIMIT=$((MEMORY_GB * 1024 * 1024))

# Apply memory limit
ulimit -v "$MEMORY_LIMIT"

# Start the Python script with CPU affinity
echo "Starting the script with CPU cores $CPU_CORES and memory limit ${MEMORY_GB}GB..."
nohup taskset -c "$CPU_CORES" python3 -u "$SCRIPT_PATH" "$@" > "$LOG_FILE" 2>&1 &

PID=$!

if [[ "$PID" =~ ^[0-9]+$ ]]; then
    mv "$LOG_FILE" "output_files/output$PID.log"
    LOG_FILE="output_files/output$PID.log"

    if ps -p $PID > /dev/null; then
        echo "Script is running with PID $PID. Check $LOG_FILE for output."
    else
        echo "Failed to start the script."
    fi
else
    echo "Failed to capture a valid PID. The script may not have started correctly."
fi
