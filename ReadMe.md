

### Run the project

To process a video, run the following command:

```bash
python .\run.py --input_video input_video_path --output_base_path output_video_path 
```

use the `--help` flag to see all the available options:

```bash
python .\run.py --help
```

### Project Structure

The main project is structured as follows:

```plaintext

├── data
├── dive-pose-estimator
├── models
    ├── mmpose
    ├── RobustVideoMatting
├── run.py
├── README.md
└── requirements.txt

```

### Data

The `data` directory contains the input video files that you want to process. You can place your video files in this directory for easy access.

### Dive Pose Estimator

The `dive-pose-estimator` directory contains the code for the keypoint generation, json handling and all other functions related to the pose estimation. This is where the main logic for processing the video frames is implemented.

### Models

This directory contains the pre-trained models used for pose estimation and video matting. The `mmpose` directory contains the models for pose estimation, while the `RobustVideoMatting` directory contains the models for segmentation and matting. These models are used to extract the keypoints and segment the video frames.

These models have to be cloned from their respective repositories and placed in the `models` directory. You can find the repositories here:

1. https://github.com/open-mmlab/mmpose - for pose estimation
2. https://github.com/PeterL1n/RobustVideoMatting - for video matting

After cloning the repositories, make sure to install the required dependencies for each model. You can do this by following the instructions in their respective README files.

Once you have cloned the repositories and installed the dependencies, you can use the models in your project by importing them in your code. 

1. Copy the bg_remove_rvm.py file from `dive-pose-estimator` directory to models/RobustVideoMatting . This file contains the code for background removal using the Robust Video Matting model.
2. Copy the pose_estimation.py file from `dive-pose-estimator` directory to models/mmpose/demo folder. This file contains the code for pose estimation using the MMPose model.

Once these files are placed in the correct directories, the run.py script will be able to access them and perform the necessary operations automatically.

### `run.py`

The `run.py` script is the main entry point for the project. It processes each stage of the video processing pipeline. 

It reads the input video files
Rotates the video and crops if argument is passed
Segments the video frames using the Robust Video Matting model
Generates keypoints using the RTMPose model in MMPose
Generates KPIs (Key Performance Indicators) for the video frames
Saves the processed video frames to a new video file

### `requirements.txt`

The `requirements.txt` file lists all the Python dependencies required to run the project. You can install them using `pip`:

```bash

pip install -r requirements.txt

```

### Conclusion

This project provides a comprehensive pipeline for processing videos using pose estimation and video matting techniques. By following the instructions in this README, you should be able to set up the project and run it successfully. If you have any questions or issues, feel free to reach out for help.