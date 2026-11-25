# Makes one video per image in images/ with a smooth zoom, gentle sway, parallax depth, vignette, film grain and a fade in.
# Optional effects (soft glow pulse, drifting haze, floating dust) are commented out: search for "TRY" and uncomment the marked lines to turn them on.
# Each video is as long as its audio clip: images/scene_1.png + audio/audio_1.wav -> videos/clip_1.mp4
# Keeps your exact pixels, no AI. Run from the documentary-content folder:
#   python src/motion_video.py
import glob
import json
import os
import re
import subprocess
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# The doc name is passed in as the first argument
doc_name = sys.argv[1]
# Only the scenes listed after the first two arguments (e.g. scene_3) are made here, the others are animated with Krea instead
parallax_scenes = sys.argv[3:]
images_folder = "images/" + doc_name
audio_folder = "audio/" + doc_name
videos_folder = "videos/" + doc_name
# Finished clips (animation + narration) go here, scenes already in this folder are not made again
clips_folder = videos_folder + "/animated_narrated_clips"
os.makedirs(clips_folder, exist_ok=True)
real_photos_folder = images_folder + "/real_photos"  # put real photos here named like scene_3.jpg (or .png / .jpeg) to overlay them on scene 3
depth_maps_folder = "depth_maps/" + doc_name    # put grayscale depth maps here named like scene_3.png (white = close, black = far) to get parallax on scene 3

# Real photo overlay settings
photo_width_fraction = 0.55      # photo is this fraction of the video width (it is also kept under 80% of the height)
photo_start_seconds = 1.0        # when the photo starts fading in
photo_screen_fraction = 0.5      # how much of the video the photo is up for (0.5 = half), counting the fade in and the dissolve
photo_fade_in_seconds = 3.0      # how long the photo takes to slowly fade in
photo_dissolve_seconds = 3.0     # how long the photo takes to dissolve back into the generated picture
photo_slow_zoom = 0.03           # the photo grows by this fraction over its time on screen
photo_border_pixels = 12         # white border around the photo
shadow_margin_pixels = 30        # how far the soft shadow spreads around the photo
shadow_strength = 0.5            # 0 = no shadow, 1 = fully black shadow
background_darken = 0.3          # how much the background dims while the photo is showing
caption_font_file = "/System/Library/Fonts/Helvetica.ttc"  # font used for the caption under the photo
caption_font_size = 28           # caption text size in pixels
caption_padding_pixels = 14      # empty space around the caption text

# The camera move for every clip. The zoom ends on each scene's x and y pixel from focal_coords.json
motion_text = "slowly zoom in, fade in from black"
focal_coords_file = images_folder + "/focal_coords.json"

frames_per_second = 24  # must match the Krea clips (24), clips with different frame rates break the final join
fade_in_seconds = 2

# Soft glow pulse settings (OFF until you uncomment its two sections): the bright areas swell and fade like light breathing
glow_threshold = 120             # only pixels brighter than this glow (lower = more of the picture glows)
glow_blur = 90                   # how wide and soft the glow is (bigger = the light spills further into the dark areas around it)
glow_strength = 2.0              # how bright the glow gets at its peak
glow_pulses_per_second = 0.15     # 0.2 is one slow breath every 5 seconds

# Drifting haze settings (OFF until you uncomment its two sections): faint mist sliding sideways, thicker toward the bottom
haze_strength = 0.3              # how thick the mist is (0 = none, 0.5 = heavy)
haze_speed_pixels = 20           # how many pixels per second it drifts
haze_color = (225, 225, 230)     # blue, green, red out of 255

# Floating dust / embers settings (OFF until you uncomment its two sections): small soft dots drifting upward
dust_count = 60                  # how many dots
dust_strength = 0.4              # how bright the dots are
dust_rise_speed = 0.02           # how fast they rise, as a fraction of the screen height per second
dust_color = (190, 225, 255)     # blue, green, red out of 255 (warm for embers, try (255, 255, 255) for white dust)

# Sway settings: tiny slow camera drift so the shot feels handheld and alive (fraction of image width)
sway_amount = 0.005

# Film grain settings: moving speckles over the whole frame
use_film_grain = True
film_grain_strength = 4          # how strong the speckles are (out of 255), try 4 (subtle) to 15 (heavy)
grain_texture_count = 8         # how many different speckle pictures to cycle through (more = less repeating, uses more memory)

# Vignette settings: darker corners that pull the eye to the middle
use_vignette = True
vignette_strength = 0.4          # 0 = no darkening, 1 = corners fully black
vignette_size = 0.3              # bigger = the dark area stays further out in the corners

# Parallax depth settings: the bottom of the image (close) slides sideways more than the top (far)
use_parallax = True
parallax_pixels = 45           # how far the closest part slides, in pixels of the original image (try 30 for subtle, 100 for strong)
parallax_vertical_pixels = 21    # how far the closest part slides up and down (0 = sideways only)
parallax_speed = 0.6             # how fast it slides back and forth
use_fake_gradient_depth = False  # when a scene has no depth map: True = use a top-to-bottom gradient (stretches the image), False = no parallax for that scene

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


def get_caption(suggestions_path, scene_number):
    # Find the "## Scene N" section in real_images_suggestions.md, then return the text of its first "caption" line (or None)
    if not os.path.exists(suggestions_path):
        return None
    suggestions_text = open(suggestions_path).read()
    scene_sections = re.split(r"^## Scene ", suggestions_text, flags=re.MULTILINE)
    for scene_section in scene_sections:
        if re.match(scene_number + r"\b", scene_section):
            caption_match = re.search(r'"caption":\s*"(.*?)"\s*$', scene_section, flags=re.MULTILINE)
            return caption_match.group(1) if caption_match else None
    return None


def add_caption_strip(framed_photo, caption):
    # Add a white strip with the caption text under the photo (text is wrapped onto several lines to fit the photo width)
    font = ImageFont.truetype(caption_font_file, caption_font_size)
    max_text_width = framed_photo.shape[1] - 2 * caption_padding_pixels
    lines = []
    current_line = ""
    for word in caption.split():
        test_line = (current_line + " " + word).strip()
        if font.getlength(test_line) > max_text_width and current_line:
            lines.append(current_line)
            current_line = word
        else:
            current_line = test_line
    lines.append(current_line)

    # Draw the lines in black on a white strip as wide as the photo
    line_height = caption_font_size + 8
    strip_height = line_height * len(lines) + 2 * caption_padding_pixels
    strip = Image.new("RGB", (framed_photo.shape[1], strip_height), (255, 255, 255))
    pen = ImageDraw.Draw(strip)
    for line_number, line in enumerate(lines):
        line_x = (framed_photo.shape[1] - font.getlength(line)) / 2
        pen.text((line_x, caption_padding_pixels + line_number * line_height), line, font=font, fill=(0, 0, 0))

    # PIL uses RGB and OpenCV uses BGR, so flip the colors before stacking the strip under the photo
    strip_bgr = np.array(strip)[:, :, ::-1]
    return np.vstack([framed_photo, strip_bgr])


def load_framed_photo(photo_path, output_width, output_height, caption=None):
    # Load the real photo, shrink it to fit the video, add a white border and the caption strip (if there is a caption)
    photo = cv2.imread(photo_path)
    photo_height, photo_width = photo.shape[:2]
    shrink = min(output_width * photo_width_fraction / photo_width, output_height * 0.8 / photo_height)
    photo = cv2.resize(photo, (round(photo_width * shrink), round(photo_height * shrink)), interpolation=cv2.INTER_AREA)
    white = (255, 255, 255)
    framed_photo = cv2.copyMakeBorder(photo, photo_border_pixels, photo_border_pixels, photo_border_pixels, photo_border_pixels, cv2.BORDER_CONSTANT, value=white)
    if caption:
        framed_photo = add_caption_strip(framed_photo, caption)
    return framed_photo


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


def make_film_grain_textures(width, height):
    # Make a few random speckle pictures once and reuse them in a loop (making new noise every frame is very slow).
    # The same value is added to all colors so it looks like film grain, not color noise.
    grain_textures = []
    for texture_number in range(grain_texture_count):
        grain = np.random.normal(0, film_grain_strength, (height, width)).astype(np.float32)
        grain_textures.append(grain[:, :, None])
    return grain_textures


def make_vignette_mask(width, height):
    # A mask that is 1 in the middle and gets darker toward the corners (made once, then multiplied onto every frame)
    horizontal_curve = cv2.getGaussianKernel(width, width * vignette_size)
    vertical_curve = cv2.getGaussianKernel(height, height * vignette_size)
    bright_in_middle = vertical_curve * horizontal_curve.T
    bright_in_middle = bright_in_middle / bright_in_middle.max()
    return (1 - vignette_strength * (1 - bright_in_middle)).astype(np.float32)


def load_depth_map(depth_map_path, image_height, image_width):
    # Read a grayscale picture where white = close and black = far, as numbers from 0 to 1 in the same size as the scene image
    depth_picture = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
    depth_picture = cv2.resize(depth_picture, (image_width, image_height))
    depth_map = depth_picture.astype(np.float32) / 255
    # Soften the edges a little (blur is sized for 4K images) so the layers do not tear apart when they slide
    return cv2.GaussianBlur(depth_map, (0, 0), 8)


def make_depth_map(image_height, image_width):
    # Fake depth: the top of the image counts as far away (0) and the bottom as close (1).
    # This suits upward-looking shots (people at the bottom, sky at the top). Swap in a real depth map image here if you ever have one.
    depth_from_top_to_bottom = np.linspace(0, 1, image_height, dtype=np.float32)
    # Squared so the top half barely moves and the bottom (the close people) moves a lot, which makes the depth easier to see
    return np.tile(depth_from_top_to_bottom[:, None] ** 2, (1, image_width))


def apply_parallax(image, depth_map, pixel_grid_x, pixel_grid_y, seconds_so_far):
    # Slide each pixel sideways and up and down by an amount that depends on how close it is, so near parts move more than far parts.
    # Sine for sideways and cosine for up and down, so together the near parts trace a slow oval instead of a straight line.
    sideways_shift = np.sin(seconds_so_far * parallax_speed) * parallax_pixels
    up_and_down_shift = np.cos(seconds_so_far * parallax_speed) * parallax_vertical_pixels
    shifted_grid_x = (pixel_grid_x - depth_map * sideways_shift).astype(np.float32)
    shifted_grid_y = (pixel_grid_y - depth_map * up_and_down_shift).astype(np.float32)
    return cv2.remap(image, shifted_grid_x, shifted_grid_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def make_glow_layer(image):
    # Keep only the bright parts of the picture and blur them into a soft halo (made once per clip)
    bright_parts = np.clip(image.astype(np.float32) - glow_threshold, 0, None)
    return cv2.GaussianBlur(bright_parts, (0, 0), glow_blur)


def add_glow_pulse(lit_image, glow_layer, seconds_so_far):
    # The halo swells and fades smoothly, like slow breathing
    pulse = 0.5 + 0.5 * np.sin(2 * np.pi * glow_pulses_per_second * seconds_so_far)
    return lit_image + glow_layer * glow_strength * pulse


def make_haze_texture(image_height, image_width):
    # A soft cloudy pattern twice as wide as the image so it can slide sideways forever (made once per clip)
    small_noise = np.random.rand(image_height // 60 + 1, image_width // 60 + 1).astype(np.float32)
    haze = cv2.resize(small_noise, (image_width, image_height))
    # Blur three copies side by side and keep the middle one, so the left and right edges blend together with no visible seam
    three_copies = np.tile(haze, (1, 3))
    three_copies = cv2.GaussianBlur(three_copies, (0, 0), 40)
    haze = three_copies[:, image_width:image_width * 2]
    haze = (haze - haze.min()) / (haze.max() - haze.min())
    return np.tile(haze, (1, 2))


def add_drifting_haze(lit_image, haze_texture, seconds_so_far):
    # Slide the haze pattern sideways, make it thicker toward the bottom, and blend it over the picture
    image_height, image_width = lit_image.shape[:2]
    slide_pixels = int(seconds_so_far * haze_speed_pixels) % image_width
    haze = haze_texture[:, slide_pixels:slide_pixels + image_width]
    thicker_toward_bottom = np.linspace(0.2, 1, image_height, dtype=np.float32)[:, None]
    haze_amount = (haze * thicker_toward_bottom * haze_strength)[:, :, None]
    return lit_image * (1 - haze_amount) + np.float32(haze_color) * haze_amount


def make_dust_particles():
    # Give every dot a random starting spot (as fractions of the screen), drift speed, size and twinkle timing (made once per clip)
    return {
        "start_x": np.random.rand(dust_count),
        "start_y": np.random.rand(dust_count),
        "sideways_speed": np.random.uniform(-0.01, 0.01, dust_count),
        "rise_speed": np.random.uniform(0.5, 1.5, dust_count) * dust_rise_speed,
        "radius": np.random.randint(1, 4, dust_count),
        "timing": np.random.rand(dust_count) * 2 * np.pi,
    }


def add_floating_dust(frame, dust_particles, seconds_so_far):
    # Draw each dot where it is right now (drifting up and wobbling sideways, wrapping around the edges), then blur so they are soft
    frame_height, frame_width = frame.shape[:2]
    dust_layer = np.zeros((frame_height, frame_width), np.float32)
    for dot in range(dust_count):
        x = (dust_particles["start_x"][dot] + dust_particles["sideways_speed"][dot] * seconds_so_far + 0.01 * np.sin(seconds_so_far * 0.7 + dust_particles["timing"][dot])) % 1
        y = (dust_particles["start_y"][dot] - dust_particles["rise_speed"][dot] * seconds_so_far) % 1
        brightness = 0.6 + 0.4 * np.sin(seconds_so_far * 1.5 + dust_particles["timing"][dot])
        cv2.circle(dust_layer, (int(x * frame_width), int(y * frame_height)), int(dust_particles["radius"][dot]), float(brightness), -1)
    dust_layer = cv2.GaussianBlur(dust_layer, (0, 0), 1.5)
    return frame + dust_layer[:, :, None] * np.float32(dust_color) * dust_strength


def make_clip(image_path, audio_path, clip_path, focus_x, focus_y, photo_path=None, depth_map_path=None, caption=None):
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

    # TRY soft glow pulse: uncomment the next line AND the "soft glow pulse" line in the frame loop below
    glow_layer = make_glow_layer(image)

    # TRY drifting haze: uncomment the next line AND the "drifting haze" line in the frame loop below
    # haze_texture = make_haze_texture(image_height, image_width)

    # TRY floating dust / embers: uncomment the next line AND the "floating dust" line in the frame loop below
    dust_particles = make_dust_particles()

    # Things the parallax and vignette effects need, made once per clip
    # Depth map: use the scene's depth map picture if there is one, otherwise the fake gradient (if allowed), otherwise no parallax
    depth_map = None
    if depth_map_path:
        depth_map = load_depth_map(depth_map_path, image_height, image_width)
    elif use_fake_gradient_depth:
        depth_map = make_depth_map(image_height, image_width)
    pixel_grid_x, pixel_grid_y = np.meshgrid(np.arange(image_width, dtype=np.float32), np.arange(image_height, dtype=np.float32))
    vignette_mask = make_vignette_mask(output_width, output_height)
    grain_textures = make_film_grain_textures(output_width, output_height)

    # Load the real photo once if this scene has one
    framed_photo = None
    if photo_path:
        framed_photo = load_framed_photo(photo_path, output_width, output_height, caption)
    photo_end_seconds = photo_start_seconds + video_seconds * photo_screen_fraction

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

        # Start every frame from the plain picture, then add the light effects on top of it
        lit_image = image.astype(np.float32)

        # TRY soft glow pulse (also uncomment its line above in the setup)
        lit_image = add_glow_pulse(lit_image, glow_layer, seconds_so_far)

        # TRY drifting haze (also uncomment its line above in the setup)
        # lit_image = add_drifting_haze(lit_image, haze_texture, seconds_so_far)

        # Parallax: slide the close parts of the image more than the far parts
        if use_parallax and depth_map is not None:
            lit_image = apply_parallax(lit_image, depth_map, pixel_grid_x, pixel_grid_y, seconds_so_far)

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
        frame = cv2.warpAffine(lit_image, move_and_scale, (output_width, output_height), flags=cv2.INTER_CUBIC)

        # TRY floating dust / embers (also uncomment its line above in the setup)
        frame = add_floating_dust(frame, dust_particles, seconds_so_far)

        # Fade in from black
        if fade_in:
            frame = frame * min(1.0, seconds_so_far / fade_in_seconds)

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

        # Final look: darker corners, then film grain on top of everything
        if use_vignette:
            frame = frame * vignette_mask[:, :, None]
        if use_film_grain:
            frame = frame + grain_textures[frame_number % grain_texture_count]

        ffmpeg.stdin.write(np.clip(frame, 0, 255).astype("uint8").tobytes())

    ffmpeg.stdin.close()
    ffmpeg.wait()


# Read the zoom end point for every scene, e.g. {"scene_1": {"x": 740, "y": 187}}
focal_coords = json.load(open(focal_coords_file))

# Go through every image: scene_1.png -> audio_1 -> clip_1.mp4 (in number order, so scene_2 comes before scene_10)
image_paths = sorted(glob.glob(images_folder + "/scene_*.png"), key=lambda path: int(re.findall(r"\d+", path)[-1]))

for image_path in image_paths:
    scene_number = re.findall(r"\d+", image_path)[-1]
    if "scene_" + scene_number not in parallax_scenes:
        continue
    audio_files = glob.glob(audio_folder + "/scene_" + scene_number + ".*")
    if not audio_files:
        print("Skipping scene", scene_number, "- no scene_" + scene_number + " in", audio_folder + "/")
        continue
    clip_path = clips_folder + "/scene_" + scene_number + ".mp4"
    # Skip scenes whose finished video is already in the animated_narrated_clips folder
    if os.path.exists(clip_path):
        print("Skipping scene", scene_number, "- video already exists")
        continue
    # Use a real photo for this scene if one exists, e.g. real_photos/scene_3.jpg
    photo_files = glob.glob(real_photos_folder + "/scene_" + scene_number + ".*")
    photo_path = photo_files[0] if photo_files else None
    # Use a depth map for this scene if one exists, e.g. depth_maps/scene_3.png
    depth_map_files = glob.glob(depth_maps_folder + "/scene_" + scene_number + ".*")
    depth_map_path = depth_map_files[0] if depth_map_files else None
    # The caption for this scene's real photo, from real_images_suggestions.md
    caption = get_caption(images_folder + "/real_images_suggestions.md", scene_number)
    # The point this scene zooms in on
    scene_focal_point = focal_coords["scene_" + scene_number]
    print("Making", clip_path, "zooming to", scene_focal_point["x"], scene_focal_point["y"], "with real photo" if photo_path else "", "with depth map" if depth_map_path else "")
    make_clip(image_path, audio_files[0], clip_path, scene_focal_point["x"], scene_focal_point["y"], photo_path, depth_map_path, caption)

print("Done! Clips saved in", clips_folder + "/")
