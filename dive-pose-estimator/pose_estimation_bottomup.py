# Copyright (c) OpenMMLab. All rights reserved.
import os
import time
from argparse import ArgumentParser

import cv2
import json_tricks as json
import mmengine

from mmpose.apis import inference_bottomup, init_model
from mmpose.structures import split_instances


def process_one_image(img,
                      pose_estimator):
    # inference a single image
    batch_results = inference_bottomup(pose_estimator, img)
    results = batch_results[0]
    return results.pred_instances


def main():
    parser = ArgumentParser()
    parser.add_argument("--base_name", type=str, required=True, help="Base name of the video file")
    base_name = parser.parse_args().base_name
    input_file = f"../../data/trimmed/{base_name}_trimmed.mp4"
    output_root = f"../../data/pose-estimated/{base_name}"
    device = 'cpu'

    config = 'configs/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_8xb10-140e_coco-512x512.py'
    checkpoint = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_8xb10-140e_coco-512x512_ac7c17bf-20221228.pth'

    mmengine.mkdir_or_exist(output_root)
    pred_save_path = f'{output_root}/results_' \
            f'{os.path.splitext(os.path.basename(input_file))[0]}.json'


    # Initialize timing variables
    frame_count = 0
    total_time = 0

    model = init_model(
        config,
        checkpoint,
        device=device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=False))))

    cap = cv2.VideoCapture(input_file)

    pred_instances_list = []
    frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        frame_idx += 1

        # Start time for this frame
        start_time = time.time()

        if not success:
            break

        pred_instances = process_one_image(frame, model)

        # save prediction results
        pred_instances_list.append(
            dict(
                frame_id=frame_idx,
                instances=split_instances(pred_instances)))
            
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
                meta_info=model.dataset_meta,
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
