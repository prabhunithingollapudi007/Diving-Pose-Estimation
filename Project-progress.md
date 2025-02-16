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
        `C:/Users/prabh/.conda/envs/openmmlab/python.exe "c:/Users/prabh/FAU/Study/MaD Project Pose Estimation/dive-pose-estimator/visualize_pose.py"`

08 Jan 2025
        Use a different video for pose estimation

        cd to dive-pose-estimator

        Step 1 - rotate the video
        `C:/Users/prabh/.conda/envs/openmmlab/python.exe "c:/Users/prabh/FAU/Study/MaD Project Pose Estimation/dive-pose-estimator/rotate_video.py"`

        Step 2 - segment the video
        `C:/Users/prabh/.conda/envs/openmmlab/python.exe "c:/Users/prabh/FAU/Study/MaD Project Pose Estimation/dive-pose-estimator/media_pipe_segment.py"`

        cd to mmpose

        Step 3 - pose estimation
        `C:/Users/prabh/.conda/envs/openmmlab/python.exe "c:/Users/prabh/FAU/Study/MaD Project Pose Estimation/models/mmpose/demo/topdown_demo_with_mmdet.py"`

        cd to dive-pose-estimator

        Step 4 - visualize the poses
        `C:/Users/prabh/.conda/envs/openmmlab/python.exe "c:/Users/prabh/FAU/Study/MaD Project Pose Estimation/dive-pose-estimator/visualize_pose.py"`

        Filtering, interpolation - linear / quadratic, aspect ratio fix, angles, side by side view 

21 Jan 2025

Pose estimation techniques on Segmented video - Jana_segmented.mp4


i) Usage of Mediapipe - poor results and very fast. 
        Mediapipe: model complexity: 2
        Confidence threshold: 0.2
        Processed 961 frames in 42.24 seconds.
        Average time per frame: 0.044 seconds (22.75 FPS)
        `C:/Users/prabh/.conda/envs/openmmlab/python.exe "c:/Users/prabh/FAU/Study/MaD Project Pose Estimation/dive-pose-estimator/media_pipe_pose_estimation.py"`

ii) Usage of Movenet - average results and very fast
        Movenet - single pose, thunder
        Confidence threshold: 0.2
        Processed 961 frames in 31.46 seconds.
        Average time per frame: 0.033 seconds (30.54 FPS)
        `C:/Users/prabh/.conda/envs/openmmlab/python.exe "c:/Users/prabh/FAU/Study/MaD Project Pose Estimation/dive-pose-estimator/movenet_pose_estimation.py"`

iii) Usage of MMPose
        RTM Pose V1
        Processed 961 frames in 700.19 seconds.
        Average time per frame: 0.729 seconds (1.37 FPS)

RTM Pose various configs 

Detection on video - Jana Segment
Bounding box stats
i) RTMDet-tiny	

Processed 961 frames in 206.84 seconds.
Average time per frame: 0.215 seconds (4.65 FPS)

ii) rtmdet_nano_320-8xb32_coco-person

Processed 961 frames in 52.85 seconds.
Average time per frame: 0.055 seconds (18.18 FPS)

iii) rtmdet_m_640-8xb32_coco-person

Processed 961 frames in 268.29 seconds.
Average time per frame: 0.279 seconds (3.58 FPS)

Pose estimation on video - Jana Segment

RTM POSE

i) rtmpose-m_8xb256-420e_body8-256x192 + rtm det nano

Processed 961 frames in 160.77 seconds.
Average time per frame: 0.167 seconds (5.98 FPS)

ii) rtmpose-l_8xb256-420e_body8-256x192 + rtm det nano

Processed 961 frames in 203.93 seconds.
Average time per frame: 0.212 seconds (4.71 FPS)

iii) rtmpose-m_8xb256-420e_body8-384x288 + rtm det nano
Processed 961 frames in 189.54 seconds.
Average time per frame: 0.197 seconds (5.07 FPS)

iv) rtmpose-l_8xb256-420e_body8-384x288.py + rtm det nano

Processed 961 frames in 249.56 seconds.
Average time per frame: 0.260 seconds (3.85 FPS)

v) rtmpose-t_8xb256-420e_body8-256x192.py + rtm det nano

Processed 961 frames in 138.73 seconds.
Average time per frame: 0.144 seconds (6.93 FPS)

vi) rtmpose-s_8xb256-420e_body8-256x192.py + rtm det nano

Processed 961 frames in 151.64 seconds.
Average time per frame: 0.158 seconds (6.34 FPS)

vii) crowdpose/rtmpose-m_8xb64-210e_crowdpose-256x192.py + rtm det nano

Processed 961 frames in 136.81 seconds.
Average time per frame: 0.142 seconds (7.02 FPS)

viii) simcc/coco/simcc_res50_8xb32-140e_coco-384x288.py + rtm det nano

Processed 961 frames in 391.41 seconds.
Average time per frame: 0.407 seconds (2.46 FPS)

ix) topdown_heatmap/crowdpose/td-hm_hrnet-w32_8xb64-210e_crowdpose-256x192.py + rtm det nano

Processed 961 frames in 212.08 seconds.
Average time per frame: 0.221 seconds (4.53 FPS)

x) crowdpose/td-hm_res101_8xb64-210e_crowdpose-320x256.py + rtm det nano

Processed 961 frames in 218.26 seconds.
Average time per frame: 0.227 seconds (4.40 FPS)


Final choices 

Elena_rotated.mp4 ================================

i) RVM for segmentation + RTM Det Nano (rtmdet_nano_320-8xb32_coco-person)  + RTM Pose (rtmpose-m_8xb256-420e_body8-384x288)

Processed 1064 frames in 137.54 seconds.
Average time per frame: 0.129 seconds (7.74 FPS)

+

Processed 1064 frames in 200.56 seconds.
Average time per frame: 0.188 seconds (5.31 FPS)

comments: Bad results during stationary poses.

ii) Mediapipe for segmentation + RTM Det Nano (rtmdet_nano_320-8xb32_coco-person)  + RTM Pose (rtmpose-m_8xb256-420e_body8-384x288)

Processed 1064 frames in 41.61 seconds.
Average time per frame: 0.039 seconds (25.57 FPS)

+ 

Processed 1064 frames in 116.04 seconds.
Average time per frame: 0.109 seconds (9.17 FPS)

comments: Very bad results during stationary poses.



Lou_rotated.mp4 ================================

i) RVM for segmentation + RTM Det Nano (rtmdet_nano_320-8xb32_coco-person)  + RTM Pose (rtmpose-m_8xb256-420e_body8-384x288)

Processed 1010 frames in 134.57 seconds.
Average time per frame: 0.133 seconds (7.51 FPS)

+

Processed 1010 frames in 190.58 seconds.
Average time per frame: 0.189 seconds (5.30 FPS)

comments: bad results during twisting poses.

ii) Mediapipe for segmentation + RTM Det Nano (rtmdet_nano_320-8xb32_coco-person)  + RTM Pose (rtmpose-m_8xb256-420e_body8-384x288)

Average time per frame: 0.041 seconds (24.42 FPS)
Output video saved at ../data/segmented/Lou_segmented.mp4

+

Processed 1010 frames in 148.40 seconds.
Average time per frame: 0.147 seconds (6.81 FPS)


comments: bad results during twisting poses.

two-person-sync_rotated.mp4 ================================

i) RVM for segmentation + RTM Det Nano (rtmdet_nano_320-8xb32_coco-person)  + RTM Pose (rtmpose-m_8xb256-420e_body8-384x288)

Processed 1328 frames in 171.00 seconds.
Average time per frame: 0.129 seconds (7.77 FPS)

+

Processed 1328 frames in 153.15 seconds.
Average time per frame: 0.115 seconds (8.67 FPS)

comments: twisting poses are detected but only for one person

ii) Mediapipe for segmentation + RTM Det Nano (rtmdet_nano_320-8xb32_coco-person)  + RTM Pose (rtmpose-m_8xb256-420e_body8-384x288)

Processed 1328 frames in 53.93 seconds.
Average time per frame: 0.041 seconds (24.62 FPS)

+

Processed 1328 frames in 153.06 seconds.
Average time per frame: 0.115 seconds (8.68 FPS)


comments: twisting poses are detected but only for one person



16 Feb 2025

Visualize pose - add bounding boxes, aspect ratio fix, side by side view