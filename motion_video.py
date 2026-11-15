# No-AI "alive" motion over YOUR image: smooth camera zoom/pan toward a point you pick, gentle sway, twinkling bright spots, fade in.
# Takes seconds and keeps your exact pixels. It cannot move individual heads (that needs AI).
# Usage: python motion_video.py sample.png "slowly zoom in, fade in from black" 6 380 420
#   (the last two numbers are optional: the x and y pixel of the image to zoom towards/away from)
import sys
import subprocess
import cv2
import numpy as np

image_path = sys.argv[1]
motion_text = sys.argv[2].lower()
video_seconds = float(sys.argv[3])

frames_per_second = 30
fade_in_seconds = 2

# Twinkle settings: how bright the spots get, how fast, and what counts as a "bright spot"
twinkle_strength = 0.35
twinkles_per_second = 0.6
bright_spot_threshold = 190

# Sway settings: tiny slow camera drift so the shot feels handheld and alive (fraction of image width)
sway_amount = 0.003

# Read the motion words from the text
zoom_out = "zoom out" in motion_text
zoom_in = "zoom in" in motion_text
pan_up = "up" in motion_text or "bottom to top" in motion_text
pan_down = "down" in motion_text or "top to bottom" in motion_text
fade_in = "fade in" in motion_text

image = cv2.imread(image_path)
image_height, image_width = image.shape[:2]

# The point to zoom towards: your x y pixel, or the middle of the image if you gave none
focus_x = float(sys.argv[4]) if len(sys.argv) > 5 else image_width / 2
focus_y = float(sys.argv[5]) if len(sys.argv) > 5 else image_height / 2

# Zoom of 1 shows the whole image, 2 shows the middle half
zoom_start, zoom_end = 1.1, 1.1
if zoom_out:
    zoom_start, zoom_end = 2.0, 1.0
if zoom_in:
    zoom_start, zoom_end = 1.0, 2.0

# A pan without any zoom needs some zoom, otherwise there is nothing to move over
if (pan_up or pan_down) and zoom_start == zoom_end:
    zoom_start, zoom_end = 1.3, 1.3

# Where the camera looks at the start and the end of the shot (x, y in pixels)
middle = (image_width / 2, image_height / 2)
camera_start = middle
camera_end = middle
if zoom_in:
    camera_end = (focus_x, focus_y)
if zoom_out:
    camera_start = (focus_x, focus_y)
if pan_up:
    camera_start, camera_end = (camera_start[0], image_height), (camera_end[0], 0)
if pan_down:
    camera_start, camera_end = (camera_start[0], 0), (camera_end[0], image_height)

# Output is 1080 tall with the same shape as the image (width must be even for the video format)
output_height = 1080
output_width = round(output_height * image_width / image_height / 2) * 2

# Find the bright spots (stars, lamps, highlights) once. The mask is soft so twinkles have no hard edges.
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
bright_spot_mask = (gray_image > bright_spot_threshold).astype(np.float32)
bright_spot_mask = cv2.GaussianBlur(bright_spot_mask, (0, 0), 3)

# Give every area its own random timing so spots twinkle one by one instead of all together
random_timing = np.random.rand(image_height // 20 + 1, image_width // 20 + 1).astype(np.float32)
random_timing = cv2.resize(random_timing, (image_width, image_height)) * 2 * np.pi

# ffmpeg reads raw frames from us and writes the mp4
ffmpeg = subprocess.Popen(
    ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{output_width}x{output_height}",
     "-r", str(frames_per_second), "-i", "-", "-pix_fmt", "yuv420p", "motion_output.mp4"],
    stdin=subprocess.PIPE)

total_frames = int(video_seconds * frames_per_second)

for frame_number in range(total_frames):
    seconds_so_far = frame_number / frames_per_second
    progress = frame_number / (total_frames - 1)
    # Very smooth start and stop (no sudden speed changes anywhere)
    smooth_progress = progress ** 3 * (progress * (progress * 6 - 15) + 10)

    # Zoom changes by a constant percentage per second, which looks steadier than adding a constant amount
    zoom = zoom_start * (zoom_end / zoom_start) ** smooth_progress

    # Move the camera by the same amount the zoom has changed, so the point you picked glides straight to the middle
    if zoom_start != zoom_end:
        camera_progress = (1 / zoom_start - 1 / zoom) / (1 / zoom_start - 1 / zoom_end)
    else:
        camera_progress = smooth_progress
    camera_x = camera_start[0] + (camera_end[0] - camera_start[0]) * camera_progress
    camera_y = camera_start[1] + (camera_end[1] - camera_start[1]) * camera_progress
    camera_x += np.sin(seconds_so_far * 0.8) * sway_amount * image_width

    # Twinkle: brighten and dim the bright spots over time
    twinkle_wave = np.sin(2 * np.pi * twinkles_per_second * seconds_so_far + random_timing)
    brightness_change = 1 + twinkle_strength * bright_spot_mask * twinkle_wave
    twinkling_image = np.clip(image * brightness_change[:, :, None], 0, 255).astype(np.float32)

    # Size of the area of the image the camera sees right now
    visible_width = image_width / zoom
    visible_height = visible_width * output_height / output_width

    # Keep the camera inside the image
    camera_x = min(max(camera_x, visible_width / 2), image_width - visible_width / 2)
    camera_y = min(max(camera_y, visible_height / 2), image_height - visible_height / 2)

    # Move and scale the visible area so it fills the output frame
    scale = output_width / visible_width
    left = camera_x - visible_width / 2
    top = camera_y - visible_height / 2
    move_and_scale = np.float32([[scale, 0, -left * scale], [0, scale, -top * scale]])
    frame = cv2.warpAffine(twinkling_image, move_and_scale, (output_width, output_height), flags=cv2.INTER_CUBIC)

    # Fade in from black
    if fade_in:
        frame = frame * min(1.0, seconds_so_far / fade_in_seconds)

    ffmpeg.stdin.write(np.clip(frame, 0, 255).astype("uint8").tobytes())

ffmpeg.stdin.close()
ffmpeg.wait()
print("Saved motion_output.mp4")
