import os
import subprocess
import sys
import tempfile
import time

import numpy
import torch
from PIL import Image
from spandrel import ModelLoader

# Usage: python video_upscaler.py <input_720p_video_path> <path_to_realesr-animevideov3.pth>
input_video_path = sys.argv[1]
model_weights_path = sys.argv[2]

# Output is saved next to the input with "_1080p" added to the name
output_video_path = input_video_path.rsplit(".", 1)[0] + "_1080p.mp4"

# Use the Mac GPU (mps) through PyTorch
gpu_device = torch.device("mps")

# Load the AI model once and put it on the GPU
upscale_model = ModelLoader().load_from_file(model_weights_path)
upscale_model.to(gpu_device)
upscale_model.eval()


def upscale_one_frame(frame_path, upscaled_frame_path):
    # Turn the jpg into a tensor with values between 0 and 1, shaped (1, 3, height, width)
    frame_image = Image.open(frame_path).convert("RGB")
    frame_array = numpy.asarray(frame_image, dtype=numpy.float32) / 255
    frame_tensor = torch.from_numpy(frame_array).permute(2, 0, 1).unsqueeze(0).to(gpu_device)

    # Run the AI upscale (this model makes the frame 4x bigger)
    with torch.no_grad():
        upscaled_tensor = upscale_model(frame_tensor)

    # Turn the tensor back into an image and shrink it to exactly 1920x1080 so the files on disk stay small
    upscaled_array = upscaled_tensor.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    upscaled_image = Image.fromarray((upscaled_array * 255).round().astype("uint8"))
    upscaled_image.resize((1920, 1080), Image.LANCZOS).save(upscaled_frame_path, quality=95)


# Read the original frame rate (like "24/1") so the rebuilt video keeps the same speed
video_fps = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "default=nw=1:nk=1", input_video_path], text=True).strip()

# Frames live on disk (not in memory) so RAM use stays low
with tempfile.TemporaryDirectory() as work_folder:
    original_frames_folder = f"{work_folder}/original"
    upscaled_frames_folder = f"{work_folder}/upscaled"
    os.makedirs(original_frames_folder)
    os.makedirs(upscaled_frames_folder)

    # Step 1: split the video into jpg frames
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", input_video_path, "-q:v", "2", f"{original_frames_folder}/%06d.jpg"], check=True)

    # Step 2: AI upscale one frame at a time and print progress
    frame_file_names = sorted(os.listdir(original_frames_folder))
    total_frames = len(frame_file_names)
    start_time = time.time()

    for frames_done, frame_file_name in enumerate(frame_file_names, start=1):
        upscale_one_frame(f"{original_frames_folder}/{frame_file_name}", f"{upscaled_frames_folder}/{frame_file_name}")

        # Estimate the time left from the average speed so far
        seconds_left = (time.time() - start_time) / frames_done * (total_frames - frames_done)
        print(f"upscaled {frames_done}/{total_frames} frames, about {seconds_left / 60:.1f} min left")

    # Step 3: rebuild the video from the upscaled frames and copy the audio from the original if there is any
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", video_fps, "-i", f"{upscaled_frames_folder}/%06d.jpg", "-i", input_video_path, "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "copy", output_video_path], check=True)

print(f"saved {output_video_path}")
