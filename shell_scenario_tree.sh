#!/bin/bash

# Script to run a Python script in the background using nohup with CPU/memory limits
# Usage: ./shell_scenario_tree.sh [--cpu CPU_CORES] [--mem MEMORY_GB] [args]

CPU_CORES="0-3"
MEMORY_GB=4
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
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

MEMORY_LIMIT=$((MEMORY_GB * 1024 * 1024))
ulimit -v "$MEMORY_LIMIT"

echo "Starting the script with CPU cores $CPU_CORES and memory limit ${MEMORY_GB}GB..."
nohup taskset -c "$CPU_CORES" python3 -u "$SCRIPT_PATH" "$@" > "$LOG_FILE" 2>&1 &

PID=$!

if [[ "$PID" =~ ^[0-9]+$ ]]; then
    mv "$LOG_FILE" "output_files/output$PID.log"
    LOG_FILE="output_files/output$PID.log"
    MEMLOG="output_files/memlog_$PID.csv"

    echo "timestamp,rss_kb" > "$MEMLOG"

    # Start memory monitor
    (
        while kill -0 "$PID" 2>/dev/null; do
            rss_kb=$(grep -i VmRSS /proc/$PID/status 2>/dev/null | awk '{print $2}')
            timestamp=$(date '+%Y-%m-%d %H:%M:%S')
            echo "$timestamp,${rss_kb:-0}" >> "$MEMLOG"
            sleep 1
        done
    ) &

    echo "Script is running with PID $PID. Output: $LOG_FILE, Memory log: $MEMLOG"
else
    echo "Failed to capture a valid PID. The script may not have started correctly."
fi
