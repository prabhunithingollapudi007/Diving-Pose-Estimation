15 Aug 2024

1. MediaPipe
Live Performance: Yes, MediaPipe is designed specifically for real-time performance. It works efficiently on CPUs and can run at high frame rates, making it ideal for live applications.
Use Case: It is well-suited for live streaming, real-time monitoring, and applications where low latency is critical.
2. OpenPose with GPU Acceleration
Live Performance: Yes, OpenPose can work in real-time if it’s run on a capable GPU. On a powerful GPU (e.g., NVIDIA GTX/RTX series), it can process video frames quickly enough for live applications.
Use Case: Suitable for scenarios where high accuracy is needed and a GPU is available, like live sports analysis or motion capture in studios.
3. TensorFlow.js (PoseNet/MoveNet)
Live Performance: Yes, TensorFlow.js, when used with models like PoseNet or MoveNet, can perform real-time pose estimation directly in the browser. It’s designed to work on live webcam feeds with low latency.
Use Case: Ideal for web-based applications where the user interacts with a camera in real-time, such as browser-based fitness apps or live video filters.
4. MoveNet
Live Performance: Yes, MoveNet is specifically designed for real-time applications and can run on CPUs, GPUs, and even mobile devices. The Lightning version is optimized for speed, making it suitable for live use.
Use Case: Excellent for mobile apps, embedded systems, or any scenario where very low latency is required, like real-time fitness tracking or live gesture control.
Key Points on Live Performance:
MediaPipe and MoveNet are the best choices if you need reliable real-time performance on a range of devices, from desktops to mobile.
OpenPose can also run live but usually requires more powerful hardware (like a good GPU) to maintain high frame rates.
TensorFlow.js is great if you're developing web-based applications that require real-time pose estimation directly in the browser.


21 Aug 2024

Creating frames from video format so that it can be used in the model.
Frames are rotated and saved.

Questions:
1. Is the camera angle going to be fixed ?
2. Background to be removed or not ?
3. How many frames per second are required ?
4. Video format to be used (input and output)?



29 aug 2024

1. Work on ideal video
2. Change brightness, contrast, saturation to highlight the person in the video
3. Remove background
4. RTMO and RTM check time complexities


04 Sep 2024

1. Drawing joints on ideal yoga video

22 Oct 2024

python demo/topdown_demo_with_mmdet.py demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth --input tests/data/posetrack18/videos/000001_mpiinew_test/000001_mpiinew_test.mp4 --output-root=vis_results/demo --show --draw-heatmap

python demo/topdown_demo_with_mmdet.py demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth --show

24 Nov 2024

cd mmpose/projects/rtmpose3d/demo
python demo/body3d_img2pose_demo.py demo/rtmdet_m_640-8xb32_coco-person.py https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth configs\rtmw3d-l_8xb64_cocktail14-384x288.py rtmw3d-l_cock14-0d4ad840_20240422.pth --input ..\..\..\..\..\data\preprocessed\Elena_rotated.mp4 --output-root ..\..\..\..\..\data\preprocessed\output --show --save-predictions --device cpu


python demo/body3d_pose_lifter_demo.py   demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py  https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth  configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py  https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth  configs/body_3d_keypoint/video_pose_lift/h36m/video-pose-lift_tcn-243frm-supv-cpn-ft_8xb128-200e_h36m.py  https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth   --input ../../data/preprocessed/Elena_rotated.mp4  --output-root  ../../data/processed/vis_result  --save-predictions


13 Dec 2024

Segmentation techniques on two-person-sync_rotated.mp4
    Total frames: 1328
    Total time: 44 seconds
    Total size: 366,605 KB (366.6 MB)

i) Usage of sam model - bad results. (may be improper implementation)
        Segment anything: vit_b
        Processed 1328 frames in 701.99 seconds.
        Average time per frame: 0.529 seconds (1.89 FPS)
        File size: 163,529 KB (163.5 MB)
        `C:/Users/prabh/.conda/envs/openmmlab/python.exe "c:/Users/prabh/FAU/Study/MaD Project Pose Estimation/models/segment-anything/sam_bg_remove.py"`

ii) Usage of RVM model - great results.
        Robust Video Matting: mobilenetv3
        Dilate mask to make the mask bigger. (value = 150)
        Processed 1328 frames in 383.41 seconds.
        Average time per frame: 0.289 seconds (3.46 FPS)
        File size: 23,224 KB (23.2 MB)
        `C:/Users/prabh/.conda/envs/openmmlab/python.exe "c:/Users/prabh/FAU/Study/MaD Project Pose Estimation/models/RobustVideoMatting/bg_remove_rvm.py"`

        Pose estimation on this video
        Processed 1328 frames in 665.78 seconds.
        Average time per frame: 0.501 seconds (1.99 FPS)
        `C:/Users/prabh/.conda/envs/openmmlab/python.exe demo/topdown_demo_with_mmdet.py`

iii) Usage of Mediapipe - good results and very fast. 
        Mediapipe: Selfie Segmentation - model selection: General model
        Dilate mask to make the mask bigger. (value = 200)
        Confidence threshold: 0.2
        Processed 1328 frames in 69.85 seconds.
        Average time per frame: 0.053 seconds (19.01 FPS)
        File size: 80,947 KB (80.9 MB)
        `C:/Users/prabh/.conda/envs/openmmlab/python.exe "c:/Users/prabh/FAU/Study/MaD Project Pose Estimation/dive-pose-estimator/media_pipe_segment.py"`

        Pose estimation on this video
        Processed 1328 frames in 612.71 seconds.
        Average time per frame: 0.461 seconds (2.17 FPS)

        `C:/Users/prabh/.conda/envs/openmmlab/python.exe demo/topdown_demo_with_mmdet.py`


iv) Usage of YOLO with deeplab V3 - good results.
        YOLOv5: yolov5s
        Processed 1328 frames in 742.18 seconds.
        Average time per frame: 0.559 seconds (1.79 FPS)
        File size: 8544 KB (8.5 MB)
        `C:/Users/prabh/FAU/Study/MaD Project Pose Estimation/.venv/Scripts/python.exe "c:/Users/prabh/FAU/Study/MaD Project Pose Estimation/dive-pose-estimator/yolo_segment.py"`

        Pose estimation on this video
        Processed 1328 frames in 550.85 seconds.
        Average time per frame: 0.415 seconds (2.41 FPS)
    

19 Dec 2024
        Evaluation of the pose estimation model on two-person-sync_rotated.mp4
        Pose estimation video settings
        fps = 25
        frame_width = 2048
        frame_height = 1152

02 Jan 2025
        Visualize the poses
        `C:/Users/prabh/.conda/envs/openmmlab/python.exe "c:/Users/prabh/FAU/Study/MaD Project Pose Estimation/dive-pose-estimator/visualize_pose.py`