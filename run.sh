#!/bin/bash

# List of base names to process
base_names=("tim_2" "lotti_1" "lotti_2" "lotti_3" "c_wassen_1" "c_wassen_2" "c_wassen_3" "e_wassen_1" "e_wassen_2" "e_wassen_3" "e_wassen_4")

# Loop through each base_name
for base_name in "${base_names[@]}"; do
    echo "Processing: $base_name"
    
    # Record overall start time
    start_time=$(date +%s)

    echo "Starting bg_remove_rvm.py at $(date)"
    step1_start=$(date +%s)
    cd models/RobustVideoMatting
    C:/Users/prabh/.conda/envs/openmmlab/python.exe -u "c:/Users/prabh/FAU/Study/MaD Project Pose Estimation/models/RobustVideoMatting/bg_remove_rvm.py" --base_name $base_name
    step1_end=$(date +%s)
    echo "Completed bg_remove_rvm.py at $(date)"
    echo "Time taken: $((step1_end - step1_start)) seconds"

    echo "Starting pose_estimation.py at $(date)"
    step2_start=$(date +%s)
    cd ../mmpose
    C:/Users/prabh/.conda/envs/openmmlab/python.exe -u "c:/Users/prabh/FAU/Study/MaD Project Pose Estimation/models/mmpose/demo/pose_estimation.py" --base_name $base_name
    step2_end=$(date +%s)
    echo "Completed pose_estimation.py at $(date)"
    echo "Time taken: $((step2_end - step2_start)) seconds"

    echo "Starting visualize_pose.py at $(date)"
    step3_start=$(date +%s)
    cd ../..
    C:/Users/prabh/.conda/envs/openmmlab/python.exe -u "c:/Users/prabh/FAU/Study/MaD Project Pose Estimation/dive-pose-estimator/visualize_pose.py" --base_name $base_name
    step3_end=$(date +%s)
    echo "Completed visualize_pose.py at $(date)"
    echo "Time taken: $((step3_end - step3_start)) seconds"

    # Record overall end time
    end_time=$(date +%s)
    echo "Total execution time for $base_name: $((end_time - start_time)) seconds"
    echo "-----------------------------------------------------------"
done