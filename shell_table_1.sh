#!/bin/bash

# Lists of possible argument values
FOLDERS=("sac-vp" "sac-fr+vp" "sac-vp+cp" "sac-ws+pc" "sac-fr+ws+pc" "sac-ws+pc+cp" "ppo-vp" "ppo-fr+vp" "ppo-vp+cp" "ppo-ws+pc" "ppo-fr+ws+pc" "ppo-ws+pc+cp")
PORTS=(4)  # You can add more ports if needed
GENS=("True" "False")

# Default GPU (if none is specified in the command)
GPU=""

# Check if GPU argument is provided
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Loop over each combination of arguments
for folder in "${FOLDERS[@]}"
do
    for ports in "${PORTS[@]}"
    do
        for gen in "${GENS[@]}"
        do
            echo "Running experiment with folder=$folder, ports=$ports, gen=$gen"

            # Pass the --gpu argument to the inner script and wait for it to finish
            if [ -n "$GPU" ]; then
                # Run the inner script synchronously
                ./shell_main.sh --gpu "$GPU" --folder "$folder" --ports "$ports" --gen "$gen"
            else
                # Run the inner script synchronously
                ./shell_main.sh --folder "$folder" --ports "$ports" --gen "$gen"
            fi

            # Wait for the last job to finish before starting the next one
            wait $!
        done
    done
done
