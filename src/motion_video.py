# Makes one video per image in images/ with a smooth zoom, gentle sway, twinkling bright spots and a fade in.
# Each video is as long as its audio clip: images/scene_1.png + audio/audio_1.wav -> videos/clip_1.mp4
# Keeps your exact pixels, no AI. Run from the documentary-content folder:
#   python src/motion_video.py
import glob
import re
import subprocess
import cv2
import numpy as np

images_folder = "images"
audio_folder = "audio"
videos_folder = "videos"
real_photos_folder = "real_photos"  # put real photos here named like scene_3.jpg (or .png / .jpeg) to overlay them on scene 3

# Real photo overlay settings
photo_width_fraction = 0.55      # photo is this fraction of the video width (it is also kept under 80% of the height)
photo_start_seconds = 1.0        # when the photo starts fading in
photo_fade_seconds = 0.5         # how long the fade in and the fade out take
photo_end_margin_seconds = 0.5   # the photo is fully gone this long before the clip ends
photo_slow_zoom = 0.03           # the photo grows by this fraction over its time on screen
photo_border_pixels = 12         # white border around the photo
shadow_margin_pixels = 30        # how far the soft shadow spreads around the photo
shadow_strength = 0.5            # 0 = no shadow, 1 = fully black shadow
background_darken = 0.3          # how much the background dims while the photo is showing

# The camera move for every clip. The x and y pixel is the spot the camera zooms towards.
motion_text = "slowly zoom in, fade in from black"
focus_x = 700
focus_y = 378

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

# Zoom of 1 shows the whole image, 2 shows the middle half
zoom_start, zoom_end = 1.1, 1.1
if zoom_out:
    zoom_start, zoom_end = 2.0, 1.0
if zoom_in:
    zoom_start, zoom_end = 1.0, 2.0

# A pan without any zoom needs some zoom, otherwise there is nothing to move over
if (pan_up or pan_down) and zoom_start == zoom_end:
    zoom_start, zoom_end = 1.3, 1.3


def get_audio_seconds(audio_path):
    # Ask ffprobe (comes with ffmpeg) how long the audio is
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", audio_path], capture_output=True, text=True)
    return float(result.stdout)


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


def make_clip(image_path, audio_path, clip_path, photo_path=None):
    video_seconds = get_audio_seconds(audio_path)

    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]

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

    # Load the real photo once if this scene has one
    framed_photo = None
    if photo_path:
        framed_photo = load_framed_photo(photo_path, output_width, output_height)
    photo_end_seconds = video_seconds - photo_end_margin_seconds

    # ffmpeg reads the raw frames from us, adds the audio file and writes the mp4
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{output_width}x{output_height}",
         "-r", str(frames_per_second), "-i", "-", "-i", audio_path, "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", clip_path],
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

        # Real photo: fades in, grows slowly, fades out. The background dims while it shows.
        if framed_photo is not None:
            fade_in_amount = (seconds_so_far - photo_start_seconds) / photo_fade_seconds
            fade_out_amount = (photo_end_seconds - seconds_so_far) / photo_fade_seconds
            photo_alpha = min(1.0, max(0.0, min(fade_in_amount, fade_out_amount)))
            if photo_alpha > 0:
                time_on_screen = (seconds_so_far - photo_start_seconds) / (photo_end_seconds - photo_start_seconds)
                photo_zoom = 1 + photo_slow_zoom * time_on_screen
                frame = frame * (1 - background_darken * photo_alpha)
                frame = overlay_photo(frame, framed_photo, photo_zoom, photo_alpha)

        ffmpeg.stdin.write(np.clip(frame, 0, 255).astype("uint8").tobytes())

    ffmpeg.stdin.close()
    ffmpeg.wait()


# Go through every image: scene_1.png -> audio_1 -> clip_1.mp4 (in number order, so scene_2 comes before scene_10)
image_paths = sorted(glob.glob(images_folder + "/scene_*.png"), key=lambda path: int(re.findall(r"\d+", path)[-1]))

for image_path in image_paths:
    scene_number = re.findall(r"\d+", image_path)[-1]
    audio_files = glob.glob(audio_folder + "/audio_" + scene_number + ".*")
    if not audio_files:
        print("Skipping scene", scene_number, "- no audio_" + scene_number + " in", audio_folder + "/")
        continue
    clip_path = videos_folder + "/clip_" + scene_number + ".mp4"
    # Use a real photo for this scene if one exists, e.g. real_photos/scene_3.jpg
    photo_files = glob.glob(real_photos_folder + "/scene_" + scene_number + ".*")
    photo_path = photo_files[0] if photo_files else None
    print("Making", clip_path, "with real photo" if photo_path else "")
    make_clip(image_path, audio_files[0], clip_path, photo_path)

print("Done! Clips saved in", videos_folder + "/")
