#!/bin/bash

# ./run_job.sh "0-23" "40G" "./my_program arg1 arg2"

CPU_RANGE=$1         # e.g. "0-23"
MEM_LIMIT=$2         # e.g. "40G"
JOB_CMD=$3           # e.g. "./my_program arg1 arg2"

GROUP_NAME="job_$$"  # Unique name for job
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

OUTPUT_LOG="$LOG_DIR/${GROUP_NAME}_output.log"
MEM_LOG="$LOG_DIR/${GROUP_NAME}_mem.log"

# Start time
START_TIME=$(date +%s)

echo "Current working directory: $(pwd)"


# Run the job using systemd-run (cgroup v2 compatible)
echo "Starting job with systemd-run..."
sudo systemd-run --unit=$GROUP_NAME \
    --property=MemoryMax=$MEM_LIMIT \
    --property=CPUAffinity=$CPU_RANGE \
    --pipe bash -c "$JOB_CMD" > "$OUTPUT_LOG" 2>&1 &

PID=$!

# Log memory usage
echo "Time(s) RSS(KB)" > "$MEM_LOG"
while kill -0 $PID 2>/dev/null; do
    NOW=$(($(date +%s) - START_TIME))
    RSS=$(ps -o rss= -p $PID --no-headers)
    echo "$NOW $RSS" >> "$MEM_LOG"
    sleep 1
done

wait $PID
echo "Job finished."
echo "Output: $OUTPUT_LOG"
echo "Memory Log: $MEM_LOG"
