import json, shutil
import subprocess, os
import sys

OUTPUT_DIR = "outputs/"
run_file = "run.py"
python_path = "python"  

def run_pipeline(input_path: str, output_video_path: str, output_json_path: str, log_file_path: str, rotate: bool,
                 stage_detection: bool, start_time:float, end_time: float, board_height:float, diver_height:float):
    
    original_stdout = sys.stdout  # Save the original stdout
    with open(log_file_path, "w") as log_file:
        class Logger:
            def write(self, message):
                sys.__stdout__.write(message)
                log_file.write(message)
            def flush(self):
                sys.__stdout__.flush()
                log_file.flush()
        sys.stdout = Logger()
    
        # Place your existing logic here
        # Save pose-estimated video to output_video_path
        # Save metrics to output_json_path as JSON

        # Clear the contents of the output directory (delete files only)
        if os.path.exists(OUTPUT_DIR):
            for file_name in os.listdir(OUTPUT_DIR):
                file_path = os.path.join(OUTPUT_DIR, file_name)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # Remove file or symbolic link
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Remove subdirectory
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")

        shutil.copy(input_path, f"{OUTPUT_DIR}/input_video.mp4")
        input_path = "dive-pose-estimator/backend/outputs/input_video.mp4"  # Adjust path if needed

        # saving the input file at this location

        # output_video_path = "dive-pose-estimator/backend/" + output_video_path  # Adjust path if needed
        # output_json_path = "dive-pose-estimator/backend/" + output_json_path
        print(f"Running pipeline on {input_path} ")
        try:
            command = [
                python_path,
                run_file,
                "--input_video",
                input_path,  # Adjust path if needed
                "--output_base_path",
                "dive-pose-estimator/backend/" + OUTPUT_DIR,  # Adjust path if needed
            ]
            if rotate:
                command.append("--rotate")
            if stage_detection:
                command.append("--stage_detection")
            if start_time is not None:
                command.extend(["--start_time", str(start_time)])
            if end_time is not None:
                command.extend(["--end_time", str(end_time)])
            if board_height is not None:
                command.extend(["--board_height", str(board_height)])
            if diver_height is not None:
                command.extend(["--diver_height", str(diver_height)])

            command = " ".join(command)  # Join the command list into a single string
            print(f"Executing command: {command}")

            # Print logs in real time
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd="../../"
            )
            for line in process.stdout:
                print(line, end="")
            process.wait()
            if process.returncode != 0:
                raise Exception(f"Pipeline failed with exit code {process.returncode}")

            print("Pipeline execution completed successfully.")
        except Exception as e:
            print(f"Error during pipeline execution: {e}")
            return e

        # shutil.copy(input_path, output_video_path)
        # read the json file
        joint_angles_file = f"{OUTPUT_DIR}/joint_angles.json"
        filtered_metrics_file = f"{OUTPUT_DIR}/filtered_metrics.json"
        video_file = f"{OUTPUT_DIR}/pose_estimation_output.webm"
        with open(joint_angles_file, "r") as f:
            joint_angles = json.load(f)
        with open(filtered_metrics_file, "r") as f:
            filtered_metrics = json.load(f)
        json.dump({
            "filtered_metrics": filtered_metrics,
            "joint_angles": joint_angles
        }, open(output_json_path, "w"))
        shutil.copy(video_file, output_video_path)
        print(f"Pipeline completed. Outputs saved to {output_video_path} and {output_json_path}")
        sys.stdout = original_stdout  # Restore the original stdout
