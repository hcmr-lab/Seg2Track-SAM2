from aux_functions import *
import cv2
import os
from glob import glob
from tqdm import tqdm

def Seg2Track(detection_file, predictor, inference_state, frames_dir, params, output_folder, sequence):
    ann_obj_id = 0  # give a unique id to each object we interact with (it can be any integers)
    class_id_match = {}  # maps the class id to the object id in the inference state
    removal_dic = {}
    class_map = {1: "Car", 2: "Pedestrian"} # Class mapping (KITTI MOTS)

    detection_threshold = params["detection_threshold"]
    addition_threshold = params["addition_threshold"]
    removal_threshold = params["removal_threshold"]
    removal_tries = params["removal_tries"]
    reprompt_threshold = params["reprompt_threshold"]
    output_tag = params["output_tag"]
    save_images = params["save_images"]
    reprompt_bool = params["reprompt_bool"]
    removal_bool = params["removal_bool"]
    dataset_type =  params["dataset_type"]
    detection_type = params["detection_type"]  # Default to "MOT" if not provided

    detections = load_detections(detection_file, detection_threshold, detection_type)   

    # Initialize the inference state with objects detected in the first frame 
    if 0 in detections:   
        for instance in detections[0]['boxes']:    
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=ann_obj_id,
                box=instance[0]
            )     
            class_id_match[ann_obj_id] = instance[2]  # store the class id for this
            removal_dic[ann_obj_id] = removal_tries
            ann_obj_id += 1 
    else:
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=ann_obj_id,
            box=[0,0,0,0]  # dummy box for the first frame
        )     
        class_id_match[ann_obj_id] = 10  # store the class id for this
        removal_dic[ann_obj_id] = removal_tries
        ann_obj_id += 1

    output_frames_dir = f"{output_folder}/{output_tag}/images/frames/{sequence}"
    output_detections_frame_dir = f"{output_folder}/{output_tag}/images_detections/frames/{sequence}"
    output_masks_frame_dir = f"{output_folder}/{output_tag}/images_masks/frames/{sequence}"
    mots_output_annotations_file = f"{output_folder}/{output_tag}/annotations_mots/Seg2Track/data/{sequence}.txt"
    mot_output_annotations_file = f"{output_folder}/{output_tag}/annotations_mot/Seg2Track/data/{sequence}.txt"

    os.makedirs(output_frames_dir, exist_ok=True)
    os.makedirs(output_detections_frame_dir, exist_ok=True)
    os.makedirs(output_masks_frame_dir, exist_ok=True)
    os.makedirs(f"{output_folder}/{output_tag}/annotations_mots/Seg2Track/data", exist_ok=True)
    os.makedirs(f"{output_folder}/{output_tag}/annotations_mot/Seg2Track/data", exist_ok=True)

    with open(mots_output_annotations_file, "w") as mots_result_file:
        with open(mot_output_annotations_file, "w") as mot_result_file:
            # run propagation throughout the video and collect the results in a dict
            for frame_idx, obj_ids, masks in predictor.propagate_in_video(inference_state):
                previous_frame_ids = inference_state['obj_ids'].copy()

                # Read the current frame
                if dataset_type == "MOT":
                    output_image = cv2.imread(os.path.join(frames_dir, f"{frame_idx+1:06d}.jpg"))

                else:  # KITTI
                    output_image = cv2.imread(os.path.join(frames_dir, f"{frame_idx:06d}.jpg"))

                # Draw detection boxes on the image
                if frame_idx in detections:
                    for box, score, class_id in detections[frame_idx]['boxes']:
                        # Draw the detection box on the image
                        if frame_idx == 0:
                            cv2.rectangle(output_image, (round(box[0]), round(box[1])), (round(box[2]), round(box[3])), (255, 0, 0), 2)
                        else:
                            cv2.rectangle(output_image, (round(box[0]), round(box[1])), (round(box[2]), round(box[3])), (15, 15, 15), 2)
                    if save_images:
                        cv2.imwrite(f"{output_detections_frame_dir}/frame_{frame_idx:04d}.png", output_image)

                # Get the masks from the inference state
                if inference_state['masks'] is not None:
                    masks_list = [(obj > 0.0).cpu().numpy() for obj in inference_state['masks'].squeeze(1)]
                else:
                    masks_list = []

                # Calculate the scores for the masks
                mask_scores={}
                for obj_id in previous_frame_ids:
                    obj_idx = predictor._obj_id_to_idx(inference_state,obj_id)
                    obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                    if frame_idx in obj_output_dict["cond_frame_outputs"]:
                        current_out = obj_output_dict["cond_frame_outputs"][frame_idx]
                    else:
                        current_out = obj_output_dict["non_cond_frame_outputs"][frame_idx]
                    
                    mask_scores[obj_id]=current_out["ious"].max().item()

                if frame_idx != 0:
                    # Create a negative mask from the previous frame's output masks and check for new objects
                    new_objects = []
                    detections_not_used = []
                    if frame_idx in detections:
                        if masks_list:
                            new_objects, detections_not_used = check_new_objects(detections[frame_idx], generate_negative_mask(masks_list, list=True), addition_threshold)
                        else:
                            new_objects = [(box, score, class_id) for box, score, class_id in detections[frame_idx]['boxes']]


                    # Add new objects based on specific frames
                    for obj in new_objects:
                        box = np.array(obj[0], dtype=np.float32)
                        cv2.rectangle(output_image, (round(box[0]), round(box[1])), (round(box[2]), round(box[3])), (255, 0, 0), 2)
                        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=frame_idx,
                            obj_id=ann_obj_id,
                            box=box
                        )     
                        predictor.propagate_object_in_video(inference_state, ann_obj_id)
                        class_id_match[ann_obj_id] = obj[2]  # store the class id for this
                        removal_dic[ann_obj_id] = removal_tries
                        ann_obj_id += 1
                    
                    # Match detections to masks from the previous frame
                    if detections_not_used:
                        mask_bbox_match = match_masks_to_boxes(masks_list, detections_not_used, previous_frame_ids)
                    else:
                        mask_bbox_match = {}

                    # If the score is below the reprompt threshold, re-prompt the model
                    for obj_id in mask_scores:
                        score = mask_scores[obj_id]

                        # Check if the object should be removed based on its score
                        if score < removal_threshold and removal_bool:
                            removal_dic[obj_id] -= 1
                            if removal_dic[obj_id] <= 0:
                                predictor.remove_object(inference_state, obj_id)
                                continue

                        if score < reprompt_threshold and reprompt_bool:
                            if obj_id in mask_bbox_match:
                                print(mask_bbox_match, obj_id, class_id_match)
                                if detections_not_used[mask_bbox_match[obj_id]][1] > addition_threshold[str(class_id_match[obj_id])]:
                                    box = detections_not_used[mask_bbox_match[obj_id]][0]
                                    cv2.rectangle(output_image, (round(box[0]), round(box[1])), (round(box[2]), round(box[3])), (0, 0, 255), 2)
                                    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                                        inference_state=inference_state,
                                        frame_idx=frame_idx,
                                        obj_id=obj_id,
                                        box=box
                                    )
                                    predictor.propagate_object_in_video(inference_state, obj_id)
                # Show results
                video_segments = {
                    out_obj_id: (inference_state['masks'][i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(previous_frame_ids)
                }
                
                # To prevent overlapping masks, create a mask to keep track of already used pixels
                used_mask = np.zeros(output_image.shape[:2], dtype=bool)
                if dataset_type == "KITTI":
                    masks_image = cv2.imread(os.path.join(frames_dir, f"{frame_idx:06d}.jpg"))
                elif dataset_type == "MOT":
                    masks_image = cv2.imread(os.path.join(frames_dir, f"{frame_idx+1:06d}.jpg"))
                

                for obj_id, mask in video_segments.items():
                    mask_bool = (mask[0] > 0)

                    # Get bounding box from mask
                    ys, xs = np.where(mask_bool)

                    if len(xs) == 0 or len(ys) == 0:
                        continue  # skip empty masks

                    bbox_left = xs.min()
                    bbox_top = ys.min()
                    bbox_right = xs.max()
                    bbox_bottom = ys.max()
                    class_id = class_id_match[obj_id]
                    score = mask_scores[obj_id]
                    
                    if class_map.get(class_id) is None:
                        continue

                    # Write results to 2DMOT format file
                    if dataset_type == "KITTI":
                        line_out = f"{frame_idx} {obj_id} {class_map.get(class_id,'Unknown')} -1 -1 -1 " \
                        f"{bbox_left} {bbox_top} {bbox_right} {bbox_bottom} -1 -1 -1 -1 -1 -1 -1 {score}"
                    elif dataset_type == "MOT":
                        line_out = f"{frame_idx+1} {obj_id} {bbox_left} {bbox_top} {bbox_right-bbox_left} {bbox_bottom-bbox_top} {score} -1 -1 -1"
                    mot_result_file.write(line_out + "\n")

                    # Remove overlap: only keep mask pixels not already used
                    mask_bool = np.logical_and(mask_bool, ~used_mask)
                    used_mask = np.logical_or(used_mask, mask_bool)

                    if not np.any(mask_bool):
                        continue

                    # OpenCV uses BGR, so green is [0,255,0]
                    color = np.array(get_light_color_from_id_hash(obj_id), dtype=np.uint8)

                    alpha = 0.7
                    overlay_frame = output_image.astype(np.float32)

                    # Blend color with image where mask is True
                    overlay_frame[mask_bool] = (
                        alpha * color + (1 - alpha) * overlay_frame[mask_bool]
                    )

                    output_image = overlay_frame.astype(np.uint8)

                    # Only mask image
                    overlay_frame2 = masks_image.astype(np.float32)

                    # Blend color with image where mask is True
                    overlay_frame2[mask_bool] = (
                        alpha * color + (1 - alpha) * overlay_frame2[mask_bool]
                    )

                    masks_image = overlay_frame2.astype(np.uint8)

                    # Write the results for this object
                    img_height, img_width = output_image.shape[:2]
                    rle_mask = binary_mask_to_kitti_rle(mask_bool)

                    if dataset_type == "KITTI":
                        mots_result_file.write(f"{frame_idx} {class_id * 1000 + obj_id} {class_id} {img_height} {img_width} {rle_mask}\n")
                    elif dataset_type == "MOT":
                        mots_result_file.write(f"{frame_idx+1} {class_id * 1000 + obj_id} {class_id} {img_height} {img_width} {rle_mask}\n")
                    
                if save_images:
                    cv2.imwrite(f"{output_frames_dir}/frame_{frame_idx:04d}.png", output_image)
                    cv2.imwrite(f"{output_masks_frame_dir}/frame_{frame_idx:04d}.png", masks_image)
    return