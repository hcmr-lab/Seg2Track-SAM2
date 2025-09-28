import hashlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from pycocotools import mask as mask_utils
import cv2
import os
from PIL import Image
import pycocotools.mask as maskUtils

# Function to set up the device for PyTorch
def setup_device():
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
    return device

# Function to calculate the percentage of the bounding box that overlaps with a mask 
def mask_bbox_intersection_percentage(mask: np.ndarray, bbox: tuple) -> float:
    """
    Calculate the percentage of the bounding box that overlaps with the mask.

    Parameters:
        mask (np.ndarray): 2D binary mask (values 0 or 1), shape (H, W)
        bbox (tuple): Bounding box in (x_min, y_min, x_max, y_max) format

    Returns:
        float: Percentage of the bounding box area covered by the mask (0.0 to 1.0)
    """
    x_min, y_min, x_max, y_max = bbox
    x_min = max(0, int(x_min))
    y_min = max(0, int(y_min))
    x_max = min(mask.shape[1], int(x_max))
    y_max = min(mask.shape[0], int(y_max))

    # Check for invalid bbox
    if x_min >= x_max or y_min >= y_max:
        return 0.0

    bbox_area = (x_max - x_min) * (y_max - y_min)
    if bbox_area == 0:
        return 0.0

    # Get cropped region from the mask
    cropped_mask = mask[y_min:y_max, x_min:x_max]
    intersection_area = np.sum(cropped_mask)

    percentage = (intersection_area / bbox_area)
    return percentage

# Function to get a consistent light color for an object ID
def get_light_color_from_id_hash(object_id):
    """
    Returns a consistent light RGB color (tuple of ints 128–255) for a given object ID.
    Uses SHA-1 hashing and maps color channels to light range.
    """
    # Get hash digest from ID
    id_bytes = str(object_id).encode('utf-8')
    hash_digest = hashlib.sha1(id_bytes).digest()

    # Use the first 3 bytes, and scale each to 128–255
    r = 64 + (hash_digest[0] % 128)
    g = 64 + (hash_digest[1] % 128)
    b = 64 + (hash_digest[2] % 128)

    return (r, g, b)

# Function to load detection data from a file
def load_detections(detection_path, detection_threshold=0.50, detection_type="MOT"):
    """
    Load detection data from a file.
    """
    with open(detection_path) as f:
        detection_data = {}
        for line in f:
            # Load differently based on dataset type
            if detection_type == "KITTI":
                data=line.split()
                frame, x_min, y_min, x_max, y_max, score, class_id = data[0:7]
            elif detection_type == "MOT":
                data=line.split(",")
                frame, _, x_min, y_min, width, height, score = data[0:7]

                frame = int(frame) - 1  # MOT frames are 1-indexed, convert to 0-indexed
                x_max = float(x_min) + float(width)
                y_max = float(y_min) + float(height)
                score = float(score)/100  # MOT scores are 0-100, convert to 0-1
                class_id = 1  # Assume single class for MOT

            # Filter by detection threshold
            if float(score) < detection_threshold:
                continue

            # Store detection data
            if int(frame) not in detection_data:
                detection_data[int(frame)] = {'boxes': [([float(x_min), float(y_min), float(x_max), float(y_max)],float(score), int(class_id))]}
            else:
                detection_data[int(frame)]['boxes'].append(([float(x_min), float(y_min), float(x_max), float(y_max)],float(score), int(class_id)))
        return detection_data

# Function to generate a negative mask from output masks    
def generate_negative_mask(output_masks, list=False):
    """
    Returns a mask where all pixels not covered by any mask in output_masks are True (negative region).
    The result is the logical NOT of the union of all masks.

    Args:
        output_masks (dict): Dictionary of 2D binary masks (dtype=bool or 0/1).

    Returns:
        np.ndarray: Negative mask (True where no mask covers, False where any mask covers).
    """
    if not list:
        masks_list = [mask for mask in output_masks.values()]
    else:
        masks_list = output_masks

    if not masks_list:
        raise ValueError("output_masks is empty.")

    union_mask = np.zeros_like(masks_list[0], dtype=bool)
    for mask in masks_list:
        union_mask = np.logical_or(union_mask, mask)
    negative_mask = np.logical_not(union_mask)
    return negative_mask

# Function to check for new objects based on intersection with a negative mask
def check_new_objects(detections, negative_mask, addition_threshold=0.50):
    """
    Check new objects based on intersection with negative mask.
    """
    new_objects = []
    detections_not_used = []
    for box, score, class_id in detections['boxes']:
        if mask_bbox_intersection_percentage(negative_mask, box) > addition_threshold[str(class_id)]:
            new_objects.append((box, score, class_id))
        else:
            detections_not_used.append((box, score, class_id))
    return new_objects, detections_not_used

# Function to compute IoU between a binary mask and a bounding box
def compute_iou(mask, box):
    """
    Compute IoU between a binary mask and a bounding box.
    """
    x1, y1, x2, y2 = box
    x_min = max(0, int(x1))
    y_min = max(0, int(y1))
    x_max = min(mask.shape[1], int(x2))
    y_max = min(mask.shape[0], int(y2))
    box_mask = np.zeros_like(mask, dtype=bool)
    box_mask[y_min:y_max, x_min:x_max] = True

    intersection = np.logical_and(mask, box_mask).sum()
    union = np.logical_or(mask, box_mask).sum()
    
    return intersection / union if union > 0 else 0.0

# Function to match masks to bounding boxes using Hungarian algorithm
def match_masks_to_boxes(masks, bboxes, mask_ids, iou_threshold=0.0):
    """
    Match masks (with IDs) to bounding boxes using Hungarian algorithm.
    Returns:
        matches: list of (mask_id, box_index)
    """
    num_masks = len(masks)
    if num_masks == 0:
        return {}
    
    num_boxes = len(bboxes)
    N = max(num_masks, num_boxes)

    cost_matrix = np.full((N, N), fill_value=1.0, dtype=np.float32)  # high cost by default

    for i in range(num_masks):
        for j in range(num_boxes):
            iou = compute_iou(masks[i], bboxes[j][0])
            cost_matrix[i, j] = 1.0 - iou  # minimize (1 - IoU)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = {}
    for i, j in zip(row_ind, col_ind):
        if i < num_masks and j < num_boxes:
            iou = 1.0 - cost_matrix[i, j]
            if iou >= iou_threshold:
                matches[mask_ids[i]] = j

    return matches

def binary_mask_to_kitti_rle(binary_mask):
    """
    Converts a binary mask to KITTI-style RLE.

    Args:
        binary_mask (np.ndarray): 2D binary mask (H, W), dtype=uint8 or bool.

    Returns:
        dict: COCO-style RLE, usable in COCO annotations.
    """
    # Ensure mask is uint8 and C-contiguous
    binary_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    
    rle = mask_utils.encode(binary_mask)
    
    # pycocotools returns 'counts' as bytes, convert to str for JSON compatibility
    rle['counts'] = rle['counts'].decode('ascii')
    
    return rle['counts']

import numpy as np

def remove_mask_overlaps(tracked_objects_dict):
    # Transform into list for sorting
    objects = [
        {
            'object_id': obj_id,
            'mask': (data['mask'] > 0).astype(np.uint8),  # Ensure mask is binary
            'score': data.get('score', 1.0),
            'class_id': data['class_id']
        }
        for obj_id, data in tracked_objects_dict.items()
    ]

    # Sort by score descending
    objects.sort(key=lambda x: -x['score'])

    used_pixels = np.zeros_like(objects[0]['mask'], dtype=bool)
    output_dict = {}

    for obj in objects:
        mask = obj['mask'].astype(bool)
        cleaned_mask = np.logical_and(mask, ~used_pixels)
        used_pixels |= cleaned_mask

        output_dict[obj['object_id']] = {
            'mask': cleaned_mask.astype(np.uint8),
            'score': obj['score'],
            'class_id': obj['class_id']
        }

    return output_dict

def add_mask_to_output_image(image, mask, obj_id):
    mask_bool = (mask > 0)

    # Only overlay if mask has any pixels
    if mask_bool.sum() == 0:
        return image

    # OpenCV uses BGR, so green is [0,255,0]
    color = np.array(get_light_color_from_id_hash(obj_id), dtype=np.uint8)

    alpha = 0.7
    overlay_frame = image.astype(np.float32)

    # Blend color with image where mask is True
    overlay_frame[mask_bool] = (
        alpha * color + (1 - alpha) * overlay_frame[mask_bool]
    )

    image = overlay_frame.astype(np.uint8)
    return image

def resize_flow(flow, target_shape):
    """
    Resize optical flow to a new shape (H, W), scaling vectors accordingly.
    """
    H_orig, W_orig = flow.shape[:2]
    H_new, W_new = target_shape

    # Resize the flow map
    resized_flow = cv2.resize(flow, (W_new, H_new), interpolation=cv2.INTER_LINEAR)

    # Scale flow vectors accordingly
    scale_x = W_new / W_orig
    scale_y = H_new / H_orig
    resized_flow[..., 0] *= scale_x
    resized_flow[..., 1] *= scale_y

    return resized_flow


def warp_logits_with_flow(prev_logits, flow):
    """
    Warps logits (or binary mask) using optical flow.

    Args:
        prev_logits (np.ndarray): 2D logits (H, W)
        flow (np.ndarray): Optical flow from t-1 to t, shape (H, W, 2)

    Returns:
        np.ndarray: Warped logits (H, W)
    """
    H, W = prev_logits.shape
    flow = resize_flow(flow, (H, W))

    # Generate grid of coordinates
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    map_x = (xx + flow[..., 0]).astype(np.float32)
    map_y = (yy + flow[..., 1]).astype(np.float32)

    # Warp logits using remap
    warped = cv2.remap(prev_logits.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped

def flow_to_color(flow):
    fx, fy = flow[..., 0], flow[..., 1]
    magnitude, angle = cv2.cartToPolar(fx, fy, angleInDegrees=True)
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = angle / 2        # Hue = direction
    hsv[..., 1] = 255              # Full saturation
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value = mag
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def tensor_to_mask_list(tensor):
    """
    Converts a tensor of shape [N, 1, H, W] to a list of N objects,
    each of shape [H, W], keeping only values > 0 (others set to 0).

    Args:
        tensor (torch.Tensor): Input tensor of shape [N, 1, H, W]

    Returns:
        List[torch.Tensor]: List of N tensors, each of shape [H, W], filtered
    """
    assert tensor.ndim == 4, "Input tensor must have 4 dimensions [N, 1, H, W]"
    assert tensor.shape[1] == 1, "Second dimension should be 1 (singleton channel)"
    tensor = tensor.squeeze(1)  # Shape: [N, H, W]
    filtered = (tensor > 0).float() * tensor  # Keep only values > 0
    return [obj.cpu().numpy() for obj in filtered]

def rle_decode(rle_str, height, width):
    """
    Decode compressed RLE (as used in KITTI MOTS) into a binary mask.
    """
    rle = {
        'counts': rle_str.encode('utf-8'),
        'size': [height, width]
    }
    return maskUtils.decode(rle)

def generate_negative_masks(annotation_file, output_dir):
    """
    Generates per-frame negative masks (all objects in that frame) and saves them as PNGs.
    """
    os.makedirs(output_dir, exist_ok=True)

    frames = {}
    with open(annotation_file, "r") as f:
        for line in f:
            time_frame, obj_id, cls_id, h, w, rle = line.strip().split(" ", 5)
            time_frame = int(time_frame)
            h, w = int(h), int(w)

            mask = rle_decode(rle, h, w)

            if time_frame not in frames:
                frames[time_frame] = np.zeros((h, w), dtype=np.uint8)
            
            frames[time_frame] |= mask

    # Save negative masks
    for frame_id, obj_mask in frames.items():
        neg_mask = np.where(obj_mask == 1, 255, 0).astype(np.uint8)
        out_path = os.path.join(output_dir, f"{frame_id:04d}.png")
        Image.fromarray(neg_mask).save(out_path)

    print(f"Saved negative masks to {output_dir}")