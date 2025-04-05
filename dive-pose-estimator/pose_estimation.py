
# Copyright (c) OpenMMLab. All rights reserved.
import os
import time
from argparse import ArgumentParser

import cv2
import json_tricks as json
import mmengine
import numpy as np

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline
from argparse import ArgumentParser

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def process_one_image(img,
                      detector,
                      pose_estimator, det_cat_id=0, bbox_thr=0.2, nms_thr=0.4):

    # predict bbox
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == det_cat_id,
                                   pred_instance.scores > bbox_thr)]
    bboxes = bboxes[nms(bboxes, nms_thr), :4]

    if len(bboxes) == 0:
        pose_results = []
    else:
        # predict keypoints
        pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    return data_samples.get('pred_instances', None)


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_video", type=str, required=True, help="input_video of the video file")
    
    parser.add_argument("--output_base_path", type=str, required=True, help="Base path for results")

    input_file = parser.parse_args().input_video
    output_base_path = parser.parse_args().output_base_path

    print(f"Input video: {input_file}")
    print(f"Output base path: {output_base_path}")

    device = 'cpu'
    det_cat_id = 0
    bbox_thr = 0.2
    nms_thr = 0.5

    det_config = "models/mmpose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py"
    det_checkpoint = "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"

    # det_config = "demo/mmdetection_cfg/rtmdet_nano_320-8xb32_coco-person.py"
    # det_checkpoint = "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth"


    # pose_config = "configs/body_2d_keypoint/topdown_heatmap/crowdpose/td-hm_res101_8xb64-210e_crowdpose-320x256.py"
    # pose_checkpoint = "https://download.openmmlab.com/mmpose/top_down/resnet/res101_crowdpose_320x256-c88c512a_20201227.pth"

    pose_config = "models/mmpose/configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-384x288.py"
    pose_checkpoint = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-384x288-65e718c4_20230504.pth"

    # pose_config = "configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py"
    # pose_checkpoint = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth"

    pred_save_path = f'{output_base_path}/predictions.json'

    # Initialize timing variables
    frame_count = 0
    total_time = 0

    # build detector
    detector = init_detector(
        det_config, det_checkpoint, device=device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # build pose estimator
    pose_estimator = init_pose_estimator(
        pose_config,
        pose_checkpoint,
        device=device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=False))))

    cap = cv2.VideoCapture(input_file)
    pred_instances_list = []

    while cap.isOpened():
        success, frame = cap.read()

        # Start time for this frame
        start_time = time.time()

        if not success:
            break

        # topdown pose estimation
        pred_instances = process_one_image(frame, detector,
                                            pose_estimator, det_cat_id, bbox_thr, nms_thr)

        # save prediction results
        pred_instances_list.append(
            dict(
                frame_id=frame_count,
                instances=split_instances(pred_instances) if pred_instances else []))

        # Calculate processing time for this frame
        frame_time = time.time() - start_time
        total_time += frame_time
        frame_count += 1

        # Print progress
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")

    cap.release()

    with open(pred_save_path, 'w') as f:
        json.dump(
            dict(
                meta_info=pose_estimator.dataset_meta,
                instance_info=pred_instances_list),
            f,
            indent='\t')
    print(f'predictions have been saved at {pred_save_path}')

    # Print overall stats
    average_time_per_frame = total_time / frame_count if frame_count > 0 else 0
    print(f"Processed {frame_count} frames in {total_time:.2f} seconds.")
    print(f"Average time per frame: {average_time_per_frame:.3f} seconds ({1/average_time_per_frame:.2f} FPS)")


if __name__ == '__main__':
    main()
