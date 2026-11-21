# TO RUN (from the documentary-content folder): python src/animate_image_with_krea.py <doc_name> <narrative_path>
# For every image in images/<doc_name>/ (scene_1.png, scene_2.png, ...) that has audio in audio/<doc_name>/:
#   1. (Krea call is commented out for now) animates the image with Krea using the scene's "* Direction:" line, the raw video is kept in videos/<doc_name>/raw/
#   2. upscales the raw video to 1080p with ffmpeg
#   3. the video is slowed down to last min(14 seconds, audio length):
#      - audio 14 seconds or shorter: one steady slow speed, the video ends when the audio ends
#      - audio longer than 14 seconds: the video eases smoothly to a stop on its last frame at 14 seconds, and that frame is held while the audio keeps going
#   4. if the scene has a real photo (images/<doc_name>/real_photos/scene_N.*) it fades in over the video, stays for half the audio,
#      and dissolves out (same look as animation_parallax_generator.py)
# The finished clip (with the audio) is saved as videos/<doc_name>/scene_N.mp4
import glob
import os
import re
import subprocess
import sys
import time
import cv2
import numpy as np
import requests
from dotenv import load_dotenv
load_dotenv()

# Krea API settings (set KREA_KEY in your .env)
API_BASE = "https://api.krea.ai"
HEADERS = {"Authorization": f"Bearer {os.environ['KREA_KEY']}"}
VIDEO_MODEL_ENDPOINT = f"{API_BASE}/generate/video/bytedance/seedance-1.0-pro-fast"
VIDEO_RESOLUTION = "720p"
PHOTO_SCENE_KREA_SECONDS = 7        # length asked from Krea for scenes with a real photo (slowed down to match the audio)
MAX_VIDEO_SECONDS = 14              # the video never plays longer than this, if the audio is longer the last frame is held
EASE_OUT_SECONDS = 5                # when the audio is longer than MAX_VIDEO_SECONDS the video slows smoothly to a stop over its last this-many seconds

# Output video settings
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
FRAMES_PER_SECOND = 24

# Real photo overlay settings (same values as animation_parallax_generator.py)
photo_width_fraction = 0.55      # photo is this fraction of the video width (it is also kept under 80% of the height)
photo_start_seconds = 1.0        # when the photo starts fading in
photo_screen_fraction = 0.5      # how much of the video the photo is up for (0.5 = half), counting the fade in and the dissolve
photo_fade_in_seconds = 3.0      # how long the photo takes to slowly fade in
photo_dissolve_seconds = 3.0     # how long the photo takes to dissolve back into the video
photo_slow_zoom = 0.03           # the photo grows by this fraction over its time on screen
photo_border_pixels = 12         # white border around the photo
shadow_margin_pixels = 30        # how far the soft shadow spreads around the photo
shadow_strength = 0.5            # 0 = no shadow, 1 = fully black shadow
background_darken = 0.3          # how much the background dims while the photo is showing

# The doc name and narrative path are passed in by the orchestrator
doc_name = sys.argv[1]
narrative_path = sys.argv[2]
# Scenes listed after the first two arguments (e.g. scene_3) use the parallax animator instead, so this script skips them
parallax_scenes = sys.argv[3:]

# Folders are found relative to this file so the script works from any directory
CONTENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
images_folder = os.path.join(CONTENT_ROOT, "images", doc_name)
real_photos_folder = os.path.join(images_folder, "real_photos")
audio_folder = os.path.join(CONTENT_ROOT, "audio", doc_name)
videos_folder = os.path.join(CONTENT_ROOT, "videos", doc_name)
raw_videos_folder = os.path.join(videos_folder, "raw")
os.makedirs(raw_videos_folder, exist_ok=True)


def get_direction_prompt(narrative_path, scene_number):
    # Find the "Scene #N" header, then return the text of the next "* Direction:" line
    found_scene = False
    with open(narrative_path) as narrative_file:
        for line in narrative_file:
            if line.strip().lower() == f"scene #{scene_number}":
                found_scene = True
            elif found_scene and line.startswith("* Direction:"):
                return line.replace("* Direction:", "").strip()


def upload_image_to_krea(image_path):
    # Upload the local file so Krea gives us a URL we can use as the start image
    with open(image_path, "rb") as image_file:
        response = requests.post(f"{API_BASE}/assets", headers=HEADERS, files={"file": (os.path.basename(image_path), image_file, "image/png")})
    return response.json()["image_url"]


def start_video_job(start_image_url, prompt, video_seconds):
    # Ask Krea to animate the image with Seedance 1.0 Pro Fast
    body = {"prompt": prompt, "start_image": start_image_url, "aspect_ratio": "16:9", "resolution": VIDEO_RESOLUTION, "duration": video_seconds}
    response = requests.post(VIDEO_MODEL_ENDPOINT, headers=HEADERS, json=body)
    return response.json()["job_id"]


def wait_for_video_url(job_id):
    # Check the job every few seconds until it is finished
    while True:
        job = requests.get(f"{API_BASE}/jobs/{job_id}", headers=HEADERS).json()
        print("job status:", job["status"])
        if job["status"] == "completed":
            return job["result"]["urls"][0]
        time.sleep(5)


def download_video(video_url, save_path):
    # Save the finished video to disk
    video_bytes = requests.get(video_url).content
    with open(save_path, "wb") as video_file:
        video_file.write(video_bytes)


def get_media_seconds(media_path):
    # Ask ffprobe (comes with ffmpeg) how long a video or audio file is
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", media_path], capture_output=True, text=True)
    return float(result.stdout.strip())


def load_framed_photo(photo_path, output_width, output_height):
    # Load the real photo, shrink it to fit the video and add a white border
    photo = cv2.imread(photo_path)
    photo_height, photo_width = photo.shape[:2]
    shrink = min(output_width * photo_width_fraction / photo_width, output_height * 0.8 / photo_height)
    photo = cv2.resize(photo, (round(photo_width * shrink), round(photo_height * shrink)), interpolation=cv2.INTER_AREA)
    white = (255, 255, 255)
    return cv2.copyMakeBorder(photo, photo_border_pixels, photo_border_pixels, photo_border_pixels, photo_border_pixels, cv2.BORDER_CONSTANT, value=white)


def overlay_photo(frame, framed_photo, photo_zoom, photo_alpha):
    # Draw the photo (with a soft shadow) in the middle of the frame. photo_alpha 0 = invisible, 1 = fully visible.
    frame_height, frame_width = frame.shape[:2]
    photo = cv2.resize(framed_photo, None, fx=photo_zoom, fy=photo_zoom, interpolation=cv2.INTER_LINEAR)
    photo_height, photo_width = photo.shape[:2]
    left = (frame_width - photo_width) // 2
    top = (frame_height - photo_height) // 2

    # Soft shadow: a blurred white box on a slightly bigger canvas, used to darken the frame behind the photo
    margin = shadow_margin_pixels
    shadow = np.zeros((photo_height + 2 * margin, photo_width + 2 * margin), np.float32)
    shadow[margin:margin + photo_height, margin:margin + photo_width] = 1
    shadow = cv2.GaussianBlur(shadow, (0, 0), margin / 2)
    shadow_area = frame[top - margin:top + photo_height + margin, left - margin:left + photo_width + margin]
    shadow_area *= (1 - shadow_strength * photo_alpha * shadow)[:, :, None]

    # Blend the photo over the frame
    photo_area = frame[top:top + photo_height, left:left + photo_width]
    photo_area[:] = photo_area * (1 - photo_alpha) + photo * photo_alpha
    return frame


def get_raw_video_seconds_at(output_seconds, raw_seconds, move_seconds, ease_out_seconds):
    # Which second of the raw video should be on screen at this moment of the output.
    # The video plays at a steady slow speed, then in the last ease_out_seconds the speed falls smoothly to zero so it glides to a stop on the last frame.
    # The steady speed is picked so the video reaches its last frame exactly when move_seconds is over (ease_out_seconds = 0 means one steady speed the whole way).
    steady_speed = raw_seconds / (move_seconds - ease_out_seconds / 2)
    steady_part_end = move_seconds - ease_out_seconds
    if output_seconds <= steady_part_end:
        return steady_speed * output_seconds
    if output_seconds < move_seconds:
        # The speed follows half a cosine wave (full speed -> zero), so it starts slowing gently and settles gently with no sudden change at either end.
        # Adding up that speed over time gives the distance travelled: (ease_out_seconds / 2) * (progress + sin(pi * progress) / pi)
        ease_out_progress = (output_seconds - steady_part_end) / ease_out_seconds
        distance_in_ease_out = steady_speed * (ease_out_seconds / 2) * (ease_out_progress + np.sin(np.pi * ease_out_progress) / np.pi)
        return steady_speed * steady_part_end + distance_in_ease_out
    return raw_seconds


def read_frame(decoder):
    # Read the next 1080p frame from ffmpeg as a float picture, or None when the video has ended
    frame_bytes = decoder.stdout.read(OUTPUT_WIDTH * OUTPUT_HEIGHT * 3)
    if len(frame_bytes) < OUTPUT_WIDTH * OUTPUT_HEIGHT * 3:
        return None
    return np.frombuffer(frame_bytes, np.uint8).reshape(OUTPUT_HEIGHT, OUTPUT_WIDTH, 3).astype(np.float32)


def make_clip(raw_video_path, audio_path, final_video_path, photo_path):
    audio_seconds = get_media_seconds(audio_path)
    raw_seconds = get_media_seconds(raw_video_path)

    # The video moves for 14 seconds at most. If the audio is longer, the video eases to a stop on its last frame and holds it until the audio ends.
    # If the audio is 14 seconds or shorter, the video is slowed to fit the audio at one steady speed with no ease out.
    move_seconds = min(MAX_VIDEO_SECONDS, audio_seconds)
    ease_out_seconds = 0
    if audio_seconds > MAX_VIDEO_SECONDS:
        ease_out_seconds = EASE_OUT_SECONDS

    # Decoder: upscale the raw video to 1080p and send us its frames at normal speed (we slow it down ourselves below)
    video_filter = f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:flags=lanczos"
    decoder = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", raw_video_path, "-vf", video_filter, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)

    # Encoder: takes our finished frames, adds the audio file and writes the mp4
    encoder = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}", "-r", str(FRAMES_PER_SECOND), "-i", "-", "-i", audio_path, "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", final_video_path], stdin=subprocess.PIPE)

    # Load the real photo once if this scene has one
    framed_photo = None
    if photo_path:
        framed_photo = load_framed_photo(photo_path, OUTPUT_WIDTH, OUTPUT_HEIGHT)
    photo_end_seconds = photo_start_seconds + audio_seconds * photo_screen_fraction

    # Only two raw frames are kept in memory at a time: the one just before the current moment and the one just after
    raw_frame_number = 0
    raw_frame = read_frame(decoder)
    next_raw_frame = read_frame(decoder)

    total_output_frames = int(audio_seconds * FRAMES_PER_SECOND)

    for frame_number in range(total_output_frames):
        seconds_so_far = frame_number / FRAMES_PER_SECOND

        # Work out where in the raw video we are (in frames, e.g. 41.3 = a bit after frame 41) and step forward to that frame
        raw_position = get_raw_video_seconds_at(seconds_so_far, raw_seconds, move_seconds, ease_out_seconds) * FRAMES_PER_SECOND
        while raw_frame_number < int(raw_position) and next_raw_frame is not None:
            raw_frame = next_raw_frame
            next_raw_frame = read_frame(decoder)
            raw_frame_number += 1

        # Blend the two neighbouring raw frames so slow motion looks smooth instead of stepping from frame to frame
        if next_raw_frame is None:
            frame = raw_frame.copy()
        else:
            blend_amount = raw_position - int(raw_position)
            frame = raw_frame * (1 - blend_amount) + next_raw_frame * blend_amount

        # Real photo: fades in, grows slowly, fades out. The background dims while it shows.
        if framed_photo is not None:
            fade_in_amount = (seconds_so_far - photo_start_seconds) / photo_fade_in_seconds
            fade_out_amount = (photo_end_seconds - seconds_so_far) / photo_dissolve_seconds
            photo_alpha = min(1.0, max(0.0, min(fade_in_amount, fade_out_amount)))
            if photo_alpha > 0:
                time_on_screen = (seconds_so_far - photo_start_seconds) / (photo_end_seconds - photo_start_seconds)
                photo_zoom = 1 + photo_slow_zoom * time_on_screen
                frame = frame * (1 - background_darken * photo_alpha)
                frame = overlay_photo(frame, framed_photo, photo_zoom, photo_alpha)

        encoder.stdin.write(np.clip(frame, 0, 255).astype("uint8").tobytes())

    encoder.stdin.close()
    encoder.wait()
    # The decoder may still have unread frames, so stop it instead of waiting for it
    decoder.kill()


# Go through every scene image in number order (so scene_2 comes before scene_10)
image_paths = glob.glob(images_folder + "/scene_*.png")
image_paths.sort(key=lambda path: int(re.findall(r"\d+", os.path.basename(path))[0]))

for image_path in image_paths:
    scene_number = re.findall(r"\d+", os.path.basename(image_path))[0]
    raw_video_path = f"{raw_videos_folder}/scene_{scene_number}.mp4"
    final_video_path = f"{videos_folder}/scene_{scene_number}.mp4"

    # Skip scenes that the parallax animator makes
    if f"scene_{scene_number}" in parallax_scenes:
        print("Skipping scene", scene_number, "- made by the parallax animator")
        continue

    # Skip scenes with no narration audio because there is nothing to match the length to
    audio_files = glob.glob(f"{audio_folder}/scene_{scene_number}.*")
    if not audio_files:
        print("Skipping scene", scene_number, "- no audio in", audio_folder)
        continue

    # Use a real photo for this scene if one exists, e.g. real_photos/scene_3.jpg
    photo_files = glob.glob(f"{real_photos_folder}/scene_{scene_number}.*")
    photo_path = photo_files[0] if photo_files else None

    # The Krea call is commented out for now because the raw videos are already made. Turn it back on to make missing raw videos.
    # Scenes with a photo ask Krea for 7 seconds, scenes without one ask for 12 seconds.
    # if not os.path.exists(raw_video_path):
    #     direction_prompt = get_direction_prompt(narrative_path, scene_number)
    #     print("Animating scene", scene_number, "with prompt:", direction_prompt)
    #     uploaded_image_url = upload_image_to_krea(image_path)
    #     job_id = start_video_job(uploaded_image_url, direction_prompt, PHOTO_SCENE_KREA_SECONDS)
    #     video_url = wait_for_video_url(job_id)
    #     download_video(video_url, raw_video_path)

    # Without Krea there is nothing to build the clip from unless the raw video is already on disk
    if not os.path.exists(raw_video_path):
        print("Skipping scene", scene_number, "- no raw video in", raw_videos_folder)
        continue

    print("Making", final_video_path, "with real photo" if photo_path else "without real photo")
    make_clip(raw_video_path, audio_files[0], final_video_path, photo_path)

print("Done! Clips saved in", videos_folder)
