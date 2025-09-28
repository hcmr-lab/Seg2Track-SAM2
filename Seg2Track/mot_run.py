from sam2.build_sam import build_sam2_video_predictor
from aux_functions import *
from Seg2Track import Seg2Track
import argparse
import json

sam2_checkpoint = "/home/diogo/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Parse command line arguments
parser = argparse.ArgumentParser(description="Load config from JSON file")
parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to the JSON configuration file"
)
args = parser.parse_args()

with open(args.config, "r") as f:
    params = json.load(f)
print("Loaded config:", params)

##################################################

subset = params["subset"]  # can be "train" or "testing"
base_path = params["base_path"]
detections_path = params["detections_path"]

for sequence in os.listdir(f"{base_path}/{subset}"):
    print(f"Sequence {sequence}")
    frames_dir = f"{base_path}/{subset}/{sequence}/img1"
    output_folder = f"{base_path}/output" # Output base directory
    detections_file = detections_path.replace("$sequence$", sequence)

    # Run Seg2Track with reprompting
    device = setup_device()
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device, offload_video_to_cpu=True, async_loading_frames=True, offload_state_to_cpu=True)
    inference_state = predictor.init_state(video_path=frames_dir)
    Seg2Track(detections_file, predictor, inference_state, frames_dir, params, output_folder, sequence)
    predictor.reset_state(inference_state)
    with torch.no_grad():
        torch.cuda.empty_cache()