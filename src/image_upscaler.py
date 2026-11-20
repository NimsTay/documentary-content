# Upscales every ~1k image in images/<doc_name>/ by 2x locally using the free Swin2SR model (downloads ~50MB on first run).
# images/yuri_gargarin/scene_3.png -> images/yuri_gargarin/upscaled/scene_3.png
# Run from the documentary-content folder:
#   python src/image_upscaler.py yuri_gargarin
import glob
import os
import sys
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

# The doc name is passed in as the first argument
doc_name = sys.argv[1]
images_folder = "images/" + doc_name
upscaled_folder = images_folder + "/upscaled"

# Images whose longest side is at most this many pixels count as "1k" images
MAX_SIDE_FOR_1K_IMAGE = 1300

os.makedirs(upscaled_folder, exist_ok=True)

# Load the 2x upscaling model once
model_name = "caidas/swin2SR-classical-sr-x2-64"
image_processor = AutoImageProcessor.from_pretrained(model_name)
upscale_model = Swin2SRForImageSuperResolution.from_pretrained(model_name)

image_paths = sorted(glob.glob(images_folder + "/scene_*.png"))

for image_path in image_paths:
    image = Image.open(image_path).convert("RGB")

    # Skip images that are already bigger than 1k
    if max(image.size) > MAX_SIDE_FOR_1K_IMAGE:
        print("Skipping", image_path, "(not a 1k image)")
        continue

    upscaled_path = upscaled_folder + "/" + os.path.basename(image_path)
    print("Upscaling", image_path, "->", upscaled_path)

    # Run the model on the image without tracking gradients (saves memory)
    model_inputs = image_processor(image, return_tensors="pt")
    with torch.no_grad():
        model_output = upscale_model(**model_inputs)

    # Turn the output tensor (values 0-1, channels first) back into a normal picture
    output_pixels = model_output.reconstruction.squeeze().clamp(0, 1).numpy()
    output_pixels = np.moveaxis(output_pixels, 0, -1)
    upscaled_image = Image.fromarray((output_pixels * 255).round().astype("uint8"))
    upscaled_image.save(upscaled_path)
