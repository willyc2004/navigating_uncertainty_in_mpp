#!/bin/bash

# ./run_job.sh "0-23" "40G" "./my_program arg1 arg2"

CPU_RANGE=$1         # e.g. "0-23"
MEM_LIMIT=$2         # e.g. "40G"
JOB_CMD=$3           # e.g. "./my_program arg1 arg2"

GROUP_NAME="job_$$"  # Unique cgroup per run, using PID

# Create cgroup and set memory
sudo cgcreate -g memory:/$GROUP_NAME
echo $MEM_LIMIT | sudo tee /sys/fs/cgroup/memory/$GROUP_NAME/memory.limit_in_bytes

# Start the job with taskset + cgroup
OUTPUT_LOG="./logs/${GROUP_NAME}_output.log"
sudo cgexec -g memory:/$GROUP_NAME taskset -c "$CPU_RANGE" bash -c "$JOB_CMD" > "$OUTPUT_LOG" 2>&1 &
PID=$!

# Log memory usage
LOG_FILE="./logs/${GROUP_NAME}_mem.log"
mkdir -p logs
echo "Time(s) RSS(KB)" > "$LOG_FILE"
START_TIME=$(date +%s)
while kill -0 $PID 2>/dev/null; do
    NOW=$(($(date +%s) - START_TIME))
    RSS=$(ps -o rss= -p $PID --no-headers)
    echo "$NOW $RSS" >> "$LOG_FILE"
    sleep 1
done

wait $PID
echo "Job finished. Log: $LOG_FILE"
