# Makes a depth map for every image in images/ using the free Depth Anything V2 Small model (runs locally, ~100MB download on first run).
# images/scene_3.png -> depth_maps/scene_3.png (white = close, black = far). motion_video.py uses these for parallax.
# Run from the documentary-content folder:
#   python src/make_depth_maps.py
import glob
import os
import re
from PIL import Image
from transformers import pipeline

images_folder = "images"
depth_maps_folder = "depth_maps"

os.makedirs(depth_maps_folder, exist_ok=True)

# Load the depth model once (it is small, so the default CPU is fast enough and avoids Mac GPU quirks)
depth_estimator = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

# Go through every image in number order (scene_2 before scene_10)
image_paths = sorted(glob.glob(images_folder + "/scene_*.png"), key=lambda path: int(re.findall(r"\d+", path)[-1]))

for image_path in image_paths:
    scene_number = re.findall(r"\d+", image_path)[-1]
    depth_map_path = depth_maps_folder + "/scene_" + scene_number + ".png"
    print("Making", depth_map_path)

    # The model returns a grayscale picture where brighter means closer
    depth_picture = depth_estimator(Image.open(image_path).convert("RGB"))["depth"]
    depth_picture.save(depth_map_path)

print("Done! Depth maps saved in", depth_maps_folder + "/")
