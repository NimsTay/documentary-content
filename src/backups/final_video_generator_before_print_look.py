# TO RUN (from the documentary-content folder):
#   preview stills: python src/final_video_generator.py <doc_name> --preview
#   full render:    python src/final_video_generator.py <doc_name>
#   pick a theme:   add --theme gallery (or card_deck, blocks, depth)
# Turns videos/<doc_name>/animated_narrated_clips/scene_1.mp4, scene_2.mp4, ... into one presentation-style video with
# an opener, section cards, animated transitions, sound design, music and captions.
# Outputs videos/<doc_name>/final_video.mp4 and final_video.srt
#
# Optional titles file inputs/<doc_name>/titles.json:
#   { "video_title": "My Video", "scene_titles": { "1": "First Topic", "4": "Next Topic" } }
# A scene with a title starts a new topic and gets a section card before it.
import argparse
import colorsys
import difflib
import json
import math
import os
import random
import re
import subprocess
import wave

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageFilter, ImageFont

# ============ SETTINGS ============
SHAPE = "16:9"                  # "16:9" or "9:16"
FRAMES_PER_SECOND = 30
AUDIO_SAMPLE_RATE = 44100
SCENE_ORDER = []                # example [1, 3, 2]. Empty means number order
CAPTIONS_ON_TOP_SCENES = []     # scene numbers that already have text at the bottom
SFX_FOLDER = ""                 # folder with whoosh.wav, hit.wav, tick.wav, riser.wav. Empty means make them with numpy
MUSIC_FILE = ""                 # your own music track. Empty means play the theme's ostinato
WHISPER_MODEL_NAME = "small"
TITLE_FONT = ""                 # a font file name in the fonts/ folder (e.g. "BebasNeue.ttf") or a full path. Empty uses the theme's font
CAPTION_FONT = ""               # same as TITLE_FONT, for captions
OPENER_SECONDS = 3.0
SECTION_CARD_SECONDS = 1.5
TRANSITION_SECONDS = 0.8
CLOSER_SECONDS = 4.0
MUSIC_VOLUME = 0.25
MUSIC_VOLUME_UNDER_VOICE = 0.1
SFX_VOLUME = 0.25
# ==================================

VIDEO_WIDTH, VIDEO_HEIGHT = (1920, 1080) if SHAPE == "16:9" else (1080, 1920)
SLIDE_DIRECTIONS = ["left", "up", "right", "down"]

project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("doc_name")
parser.add_argument("narrative_path", nargs="?")
parser.add_argument("--preview", action="store_true")
parser.add_argument("--theme")
parser.add_argument("--title-font")
parser.add_argument("--caption-font")
arguments = parser.parse_args()

doc_name = arguments.doc_name
narrative_path = arguments.narrative_path or os.path.join(project_folder, "inputs", doc_name, "narrative.txt")
videos_folder = os.path.join(project_folder, "videos", doc_name)
# The animators save their finished scene clips here
clips_folder = os.path.join(videos_folder, "animated_narrated_clips")
build_folder = os.path.join(videos_folder, "build")
preview_folder = os.path.join(build_folder, "preview")
os.makedirs(preview_folder, exist_ok=True)


# ---------- theme ----------
# --preview picks a new random theme (run it again to reroll). The full render reuses the previewed theme.
def choose_theme_name():
    theme_choice_path = os.path.join(build_folder, "theme_choice.txt")
    all_theme_names = [name[:-5] for name in os.listdir(os.path.join(project_folder, "themes")) if name.endswith(".json")]
    if arguments.theme:
        theme_name = arguments.theme
    elif os.path.exists(theme_choice_path) and not arguments.preview:
        theme_name = open(theme_choice_path).read()
    else:
        theme_name = random.choice(all_theme_names)
    open(theme_choice_path, "w").write(theme_name)
    return theme_name


theme_name = choose_theme_name()
theme = json.load(open(os.path.join(project_folder, "themes", theme_name + ".json")))
print("Theme:", theme_name)


# ---------- random colours ----------
# --preview makes new random colours every time. The full render reuses the previewed colours.
def hsv_to_hex(hue, saturation, value):
    red, green, blue = colorsys.hsv_to_rgb(hue % 1, saturation, value)
    return "#%02x%02x%02x" % (int(red * 255), int(green * 255), int(blue * 255))


def choose_colors():
    colors_path = os.path.join(build_folder, "colors.json")
    if os.path.exists(colors_path) and not arguments.preview:
        return json.load(open(colors_path))
    background_hue = random.random()
    # the accent sits roughly across the colour wheel so it stands out
    accent_hue = background_hue + random.uniform(0.35, 0.65)
    if random.random() < 0.5:
        # light paper look
        colors = {"background": hsv_to_hex(background_hue, random.uniform(0.06, 0.15), 0.94), "text": hsv_to_hex(background_hue, 0.5, 0.15), "accent": hsv_to_hex(accent_hue, 0.85, 0.85)}
    else:
        # dark chalkboard look
        colors = {"background": hsv_to_hex(background_hue, random.uniform(0.3, 0.5), 0.12), "text": hsv_to_hex(background_hue, 0.05, 0.97), "accent": hsv_to_hex(accent_hue, 0.8, 0.98)}
    json.dump(colors, open(colors_path, "w"))
    return colors


colors = choose_colors()
background_color = colors["background"]
text_color = colors["text"]
accent_color = colors["accent"]
print("Colors:", colors)


# font order: command line flag, then the setting at the top, then the theme file.
# A plain file name is looked up in the fonts/ folder, a full path is used as it is
def load_font(font_kind, size):
    chosen_fonts = {"title": arguments.title_font or TITLE_FONT, "caption": arguments.caption_font or CAPTION_FONT}
    font_name = chosen_fonts[font_kind] or theme["fonts"][font_kind]
    font_path = font_name if os.path.isabs(font_name) else os.path.join(project_folder, "fonts", font_name)
    return ImageFont.truetype(font_path, size)


# font sizes are based on the shorter side so both shapes look right
base_size = min(VIDEO_WIDTH, VIDEO_HEIGHT)
big_title_font = load_font("title", int(base_size * 0.10))
number_font = load_font("title", int(base_size * 0.18))
caption_font = load_font("caption", int(base_size * 0.05))
label_font = load_font("title", int(base_size * 0.035))


# ---------- easing ----------
def clamp_zero_to_one(value):
    return max(0.0, min(1.0, value))


def ease_in_out(progress):
    return progress * progress * (3 - 2 * progress)


# goes a little past 1 and settles back, for things that "land"
def ease_with_overshoot(progress):
    shifted = progress - 1
    return 1 + 2.7 * shifted ** 3 + 1.7 * shifted ** 2


# ---------- reading clips ----------
def get_clip_seconds(clip_path):
    output = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", clip_path])
    return float(output)


# ffmpeg filter that fits any clip into the video size and frame rate
fit_filter = f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:force_original_aspect_ratio=decrease,pad={VIDEO_WIDTH}:{VIDEO_HEIGHT}:(ow-iw)/2:(oh-ih)/2,fps={FRAMES_PER_SECOND}"


# gives the clip's frames one at a time so memory stays low
def read_clip_frames(clip_path, start_seconds=0):
    command = ["ffmpeg", "-loglevel", "error", "-ss", str(start_seconds), "-i", clip_path, "-vf", fit_filter, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    bytes_per_frame = VIDEO_WIDTH * VIDEO_HEIGHT * 3
    while True:
        frame_bytes = process.stdout.read(bytes_per_frame)
        if len(frame_bytes) < bytes_per_frame:
            break
        yield Image.frombytes("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), frame_bytes)
    process.wait()


def read_first_frame(clip_path):
    frames = read_clip_frames(clip_path)
    first_frame = next(frames)
    for _ in frames:  # drain the rest so ffmpeg closes cleanly
        pass
    return first_frame


def read_last_frame(clip_path, clip_seconds):
    for frame in read_clip_frames(clip_path, max(0, clip_seconds - 0.5)):
        last_frame = frame
    return last_frame


# zooms, tilts (degrees) and shifts a whole frame like a camera. warpAffine allows tiny sub-pixel steps so it does not jitter
def camera_move(image, zoom, angle, shift_x=0, shift_y=0):
    camera_matrix = cv2.getRotationMatrix2D((VIDEO_WIDTH / 2, VIDEO_HEIGHT / 2), angle, zoom)
    camera_matrix[0, 2] += shift_x
    camera_matrix[1, 2] += shift_y
    return Image.fromarray(cv2.warpAffine(np.array(image), camera_matrix, (VIDEO_WIDTH, VIDEO_HEIGHT), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))


# every scene gets a different camera move, in this order, so no two scenes in a row move the same way.
# All moves stay zoomed in a little so the edges of the clip never show
SCENE_MOVES = ["push_in", "pan_right", "pull_out", "pan_left", "tilt_drift"]


def move_scene_camera(frame, progress, scene_index):
    move_name = SCENE_MOVES[scene_index % len(SCENE_MOVES)]
    eased = ease_in_out(progress)
    if move_name == "push_in":
        return camera_move(frame, 1.04 + 0.08 * eased, 0)
    if move_name == "pull_out":
        return camera_move(frame, 1.12 - 0.08 * eased, 0)
    if move_name == "pan_right":
        return camera_move(frame, 1.1, 0, (0.5 - eased) * VIDEO_WIDTH * 0.06)
    if move_name == "pan_left":
        return camera_move(frame, 1.1, 0, (eased - 0.5) * VIDEO_WIDTH * 0.06)
    return camera_move(frame, 1.08 + 0.03 * eased, -1.5 + 3 * eased, 0, (0.5 - eased) * VIDEO_HEIGHT * 0.04)


# ---------- find the scenes ----------
clip_names = [name for name in os.listdir(clips_folder) if re.fullmatch(r"scene_\d+\.mp4", name)]
scene_numbers = SCENE_ORDER or sorted(int(re.findall(r"\d+", name)[0]) for name in clip_names)

titles_path = os.path.join(project_folder, "inputs", doc_name, "titles.json")
titles = json.load(open(titles_path)) if os.path.exists(titles_path) else {}
video_title = titles.get("video_title", doc_name.replace("_", " ").title())
scene_titles = titles.get("scene_titles", {})

print("Reading", len(scene_numbers), "clips...")
scenes = []
for scene_number in scene_numbers:
    clip_path = os.path.join(clips_folder, f"scene_{scene_number}.mp4")
    clip_seconds = get_clip_seconds(clip_path)
    first_frame = read_first_frame(clip_path)
    # the frozen last frame is taken at the end of this scene's camera move so it matches
    last_frame = move_scene_camera(read_last_frame(clip_path, clip_seconds), 1, len(scenes))
    scenes.append({
        "number": scene_number,
        "path": clip_path,
        "seconds": clip_seconds,
        "title": scene_titles.get(str(scene_number)),
        # full size frames are kept compressed so memory stays low
        "first_frame": cv2.imencode(".png", np.array(first_frame))[1],
        "last_frame": cv2.imencode(".png", np.array(last_frame))[1],
        # small copy used by the opener, the depth theme and the closer
        "thumbnail": first_frame.resize((VIDEO_WIDTH // 4, VIDEO_HEIGHT // 4)),
    })


def unpack_frame(compressed_frame):
    return Image.fromarray(cv2.imdecode(compressed_frame, cv2.IMREAD_COLOR))


# ---------- drawing ----------
# made once: background colour with faint graph paper lines, paper grain and darker edges
def make_paper_background():
    paper = np.full((VIDEO_HEIGHT, VIDEO_WIDTH, 3), ImageColor.getrgb(background_color), dtype=np.float32)
    text_rgb = np.array(ImageColor.getrgb(text_color), dtype=np.float32)
    grid_spacing = base_size // 18
    paper[::grid_spacing, :] = paper[::grid_spacing, :] * 0.92 + text_rgb * 0.08
    paper[:, ::grid_spacing] = paper[:, ::grid_spacing] * 0.92 + text_rgb * 0.08
    paper += np.random.normal(0, 6, (VIDEO_HEIGHT, VIDEO_WIDTH, 1))
    row_positions, column_positions = np.mgrid[0:VIDEO_HEIGHT, 0:VIDEO_WIDTH]
    distance_from_center = (row_positions / VIDEO_HEIGHT - 0.5) ** 2 + (column_positions / VIDEO_WIDTH - 0.5) ** 2
    paper *= (1 - 0.4 * distance_from_center)[:, :, None]
    return Image.fromarray(np.clip(paper, 0, 255).astype(np.uint8))


paper_background = make_paper_background()


def new_background():
    return paper_background.copy()


def split_into_lines(text, font, max_width):
    lines = [""]
    for word in text.split():
        longer_line = (lines[-1] + " " + word).strip()
        if font.getlength(longer_line) > max_width and lines[-1]:
            lines.append(word)
        else:
            lines[-1] = longer_line
    return lines


# ---------- hand drawn storyboard look ----------
# changes every few frames so drawn lines "boil" like hand animation. The render loop updates it
boil_step = 0


# moves each point a little. The same shape_id and boil_step always give the same shake
def shake_points(points, amount, shape_id):
    shaker = random.Random(shape_id * 7919 + boil_step)
    return [(x + shaker.uniform(-amount, amount), y + shaker.uniform(-amount, amount)) for x, y in points]


# a wobbly pencil line drawn twice. drawn_amount 0 to 1 draws it on from start to end
def sketch_line(draw, start, end, color, width, shape_id, drawn_amount=1.0):
    drawn_amount = clamp_zero_to_one(drawn_amount)
    if drawn_amount == 0:
        return
    points = []
    for piece_index in range(13):
        along = drawn_amount * piece_index / 12
        points.append((start[0] + (end[0] - start[0]) * along, start[1] + (end[1] - start[1]) * along))
    for pass_number in range(2):
        draw.line(shake_points(points, width * 0.7, shape_id * 3 + pass_number), fill=color, width=width, joint="curve")


# four sketchy sides drawn one after another
def sketch_rectangle(draw, box, color, width, shape_id, drawn_amount=1.0):
    left, top, right, bottom = box
    corners = [(left, top), (right, top), (right, bottom), (left, bottom), (left, top)]
    for side_index in range(4):
        side_amount = drawn_amount * 4 - side_index
        sketch_line(draw, corners[side_index], corners[side_index + 1], color, width, shape_id * 10 + side_index, side_amount)


# a quick pen circle that goes a little more than once around
def sketch_circle(draw, center, radius_x, radius_y, color, width, shape_id, drawn_amount=1.0):
    drawn_amount = clamp_zero_to_one(drawn_amount)
    if drawn_amount == 0:
        return
    point_count = 40
    points = []
    for point_index in range(point_count):
        angle = -2.2 + 2 * math.pi * 1.1 * drawn_amount * point_index / (point_count - 1)
        points.append((center[0] + radius_x * math.cos(angle), center[1] + radius_y * math.sin(angle)))
    draw.line(shake_points(points, width * 0.8, shape_id), fill=color, width=width, joint="curve")


# a sketchy line with an arrow head that appears at the end
def sketch_arrow(draw, start, end, color, width, shape_id, drawn_amount=1.0):
    sketch_line(draw, start, end, color, width, shape_id, drawn_amount / 0.8)
    if drawn_amount < 0.8:
        return
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head_length = width * 6 * clamp_zero_to_one((drawn_amount - 0.8) / 0.2)
    for side in [-1, 1]:
        head_end = (end[0] - head_length * math.cos(angle + side * 0.5), end[1] - head_length * math.sin(angle + side * 0.5))
        sketch_line(draw, end, head_end, color, width, shape_id * 5 + side + 2)


# a see-through marker stripe that wipes on from the left
def draw_highlighter(image, box, color, drawn_amount):
    left, top, right, bottom = [int(value) for value in box]
    stripe_width = int((right - left) * clamp_zero_to_one(drawn_amount))
    if stripe_width <= 0:
        return
    stripe = Image.new("RGB", (stripe_width, bottom - top), color)
    image.paste(stripe, (left, top), Image.new("L", (stripe_width, bottom - top), 140))


# marker stripe behind the lower half of each title line. Call before drawing the title words
def draw_title_highlight(image, text, center_y, seconds):
    for line_index, line in enumerate(split_into_lines(text, big_title_font, VIDEO_WIDTH * 0.85)):
        line_width = big_title_font.getlength(line)
        left = (VIDEO_WIDTH - line_width) / 2 - 20
        top = center_y + line_index * big_title_font.size * 1.2 + big_title_font.size * 0.55
        draw_highlighter(image, [left, top, left + line_width + 40, top + big_title_font.size * 0.5], accent_color, (seconds - 0.3 - line_index * 0.3) / 0.6)


def get_title_bottom(text, center_y):
    line_count = len(split_into_lines(text, big_title_font, VIDEO_WIDTH * 0.85))
    return center_y + line_count * big_title_font.size * 1.2


# a strip of see-through tape
def draw_tape(image, center, angle):
    tape_strip = Image.new("RGBA", (int(base_size * 0.09), int(base_size * 0.03)), (255, 255, 235, 150))
    tape_strip = tape_strip.rotate(angle, expand=True)
    image.paste(tape_strip, (int(center[0] - tape_strip.width / 2), int(center[1] - tape_strip.height / 2)), tape_strip)


# ---------- halftone, engraving, hatching and paper cutouts ----------
CREAM_PAPER = np.array([242, 232, 213], dtype=np.float32)
ENGRAVING_INK = np.array([45, 35, 30], dtype=np.float32)


# returns white paper with one black dot per cell. Darker parts of the picture get bigger dots
def make_halftone_mask(gray_array, dot_spacing):
    height, width = gray_array.shape
    dots = np.full((height, width), 255, dtype=np.uint8)
    for top in range(0, height, dot_spacing):
        for left in range(0, width, dot_spacing):
            darkness = 1 - gray_array[top:top + dot_spacing, left:left + dot_spacing].mean() / 255
            radius = int(round(dot_spacing * 0.7 * darkness))
            if radius > 0:
                cv2.circle(dots, (left + dot_spacing // 2, top + dot_spacing // 2), radius, 0, -1, cv2.LINE_AA)
    return dots


# turns a real photo into engraved line art on cream paper, with halftone dots in the middle tones
def make_engraving(picture):
    gray = cv2.GaussianBlur(np.array(picture.convert("L"), dtype=np.float32), (0, 0), 1.2)
    darkness = 1 - gray / 255
    rows, columns = np.mgrid[0:gray.shape[0], 0:gray.shape[1]]
    line_spacing = 4
    # wavy lines that bend with the brightness, like an engraving tool following the shapes
    line_wave = rows + 3 * np.sin(columns / 25) + gray / 255 * 6
    engraved_lines = (np.sin(2 * np.pi * line_wave / line_spacing) + 1) / 2 < darkness * 0.9
    # diagonal cross hatching only in the darkest parts
    cross_lines = (np.sin(2 * np.pi * (rows + columns) / line_spacing) + 1) / 2 < (darkness - 0.55) * 1.5
    halftone_dots = make_halftone_mask(gray.astype(np.uint8), 6) < 128
    ink_amount = np.maximum((engraved_lines | cross_lines) * 1.0, halftone_dots * 0.35)[:, :, None]
    engraving = CREAM_PAPER * (1 - ink_amount) + ENGRAVING_INK * ink_amount
    # keep 40% of the real photo so the effect is lighter
    return Image.blend(picture.convert("RGB"), Image.fromarray(engraving.astype(np.uint8)), 0.6)


# full screen of wobbly diagonal pencil lines, used as a mask for hatched shading
def make_hatch_texture(line_spacing):
    rows, columns = np.mgrid[0:VIDEO_HEIGHT, 0:VIDEO_WIDTH]
    line_position = (columns + rows + 2 * np.sin(rows / 30)) % line_spacing
    return Image.fromarray(np.where(line_position < 2.5, 255, 0).astype(np.uint8))


hatch_texture = make_hatch_texture(14)


# ---------- full screen halftone + hatching look for scene frames ----------
HALFTONE_DOT_SPACING = 8
# how far each pixel is from the middle of its halftone cell, made once
pixel_rows, pixel_columns = np.mgrid[0:VIDEO_HEIGHT, 0:VIDEO_WIDTH]
distance_to_dot_center = np.hypot(pixel_rows % HALFTONE_DOT_SPACING - HALFTONE_DOT_SPACING / 2, pixel_columns % HALFTONE_DOT_SPACING - HALFTONE_DOT_SPACING / 2).astype(np.float32)
hatch_lines_one_way = np.array(hatch_texture) > 0
hatch_lines_other_way = np.fliplr(hatch_lines_one_way)
del pixel_rows, pixel_columns


# turns a full size frame into a print: halftone dots, hatching in the shadows, cream paper with a hint of the real colour
def make_halftone_hatching(picture):
    gray = np.array(picture.convert("L"), dtype=np.float32)
    small_gray = cv2.resize(gray, (VIDEO_WIDTH // HALFTONE_DOT_SPACING, VIDEO_HEIGHT // HALFTONE_DOT_SPACING), interpolation=cv2.INTER_AREA)
    cell_darkness = cv2.resize(1 - small_gray / 255, (VIDEO_WIDTH, VIDEO_HEIGHT), interpolation=cv2.INTER_NEAREST)
    halftone_dots = distance_to_dot_center < HALFTONE_DOT_SPACING * 0.7 * cell_darkness
    pixel_darkness = 1 - gray / 255
    hatching = (hatch_lines_one_way & (pixel_darkness > 0.45)) | (hatch_lines_other_way & (pixel_darkness > 0.7))
    ink = (halftone_dots | hatching)[:, :, None]
    paper = CREAM_PAPER * 0.75 + np.array(picture, dtype=np.float32) * 0.25
    return Image.fromarray(np.where(ink, ENGRAVING_INK, paper).astype(np.uint8))


# amount 1 is the full print look, 0 is the real frame
def stylize_frame(picture, amount):
    if amount <= 0:
        return picture
    return Image.blend(picture, make_halftone_hatching(picture), clamp_zero_to_one(amount))


# fills a box with hatching lines that wipe on from the left
def draw_hatched_box(image, box, color, drawn_amount):
    left, top, right, bottom = box
    shape_mask = Image.new("L", (VIDEO_WIDTH, VIDEO_HEIGHT), 0)
    ImageDraw.Draw(shape_mask).rectangle([left, top, left + (right - left) * clamp_zero_to_one(drawn_amount), bottom], fill=255)
    image.paste(color, (0, 0), ImageChops.multiply(hatch_texture, shape_mask))


# adds a white scissor edge with a noisy wobble and a soft shadow (5, 7) px, blurred 6 px, at 32%
def make_cutout(piece):
    padding = 30
    piece_alpha = np.pad(np.array(piece.split()[3]), padding)
    grown_edge = cv2.GaussianBlur(cv2.dilate(piece_alpha, np.ones((13, 13), np.uint8)).astype(np.float32), (0, 0), 2)
    # smooth random noise pushes the edge in and out, like cutting by hand
    edge_noise = cv2.GaussianBlur(np.random.uniform(0, 1, grown_edge.shape).astype(np.float32), (0, 0), 3)
    edge_noise = (edge_noise - edge_noise.mean()) / edge_noise.std() * 60
    edge_alpha = np.maximum(np.where(grown_edge + edge_noise > 128, 255, 0), piece_alpha).astype(np.uint8)
    canvas_size = (piece_alpha.shape[1], piece_alpha.shape[0])
    shadow_alpha = Image.fromarray(np.roll(edge_alpha, (7, 5), axis=(0, 1))).filter(ImageFilter.GaussianBlur(6)).point(lambda alpha: int(alpha * 0.32))
    cutout = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    cutout.paste((0, 0, 0, 255), (0, 0), shadow_alpha)
    cutout.alpha_composite(Image.merge("RGBA", [Image.new("L", canvas_size, 255)] * 3 + [Image.fromarray(edge_alpha)]))
    cutout.alpha_composite(piece, (padding, padding))
    return cutout


# pastes a cutout so its middle sits on center, tilted and scaled
def paste_cutout(image, cutout, center, angle, width=None):
    if width:
        cutout = cutout.resize((int(width), int(width * cutout.height / cutout.width)))
    cutout = cutout.rotate(angle, expand=True, resample=Image.BICUBIC)
    image.paste(cutout, (int(center[0] - cutout.width / 2), int(center[1] - cutout.height / 2)), cutout)
    return cutout


# an engraved photo cutout, tilted, with a piece of tape on top
def paste_taped_photo(image, photo_cutout, center, photo_width, angle):
    placed = paste_cutout(image, photo_cutout, center, angle, photo_width)
    draw_tape(image, (center[0], center[1] - placed.height * 0.42), -angle * 2)


# every scene gets an engraved copy of its thumbnail and an engraved paper cutout of it
print("Making engraved cutouts...")
for scene in scenes:
    scene["engraved"] = make_engraving(scene["thumbnail"])
    scene["engraved_cutout"] = make_cutout(scene["engraved"].convert("RGBA"))


# small pluses, circles, stars and squiggles scattered near the edges, drawn on one by one
def draw_doodles(draw, shape_id, doodle_count, drawn_amount):
    placer = random.Random(shape_id)
    doodle_size = base_size * 0.025
    for doodle_index in range(doodle_count):
        # keep doodles in the side and top bands so they stay off the title and pictures
        if placer.random() < 0.5:
            x = VIDEO_WIDTH * placer.choice([placer.uniform(0.03, 0.12), placer.uniform(0.88, 0.97)])
            y = VIDEO_HEIGHT * placer.uniform(0.08, 0.92)
        else:
            x = VIDEO_WIDTH * placer.uniform(0.1, 0.9)
            y = VIDEO_HEIGHT * placer.uniform(0.04, 0.12)
        doodle_kind = placer.choice(["plus", "circle", "star", "squiggle"])
        color = accent_color if doodle_index % 2 == 0 else text_color
        amount = drawn_amount * doodle_count - doodle_index
        doodle_id = shape_id * 100 + doodle_index
        if doodle_kind == "plus":
            sketch_line(draw, (x - doodle_size, y), (x + doodle_size, y), color, 4, doodle_id, amount)
            sketch_line(draw, (x, y - doodle_size), (x, y + doodle_size), color, 4, doodle_id + 50, amount - 0.5)
        elif doodle_kind == "circle":
            sketch_circle(draw, (x, y), doodle_size * 0.7, doodle_size * 0.7, color, 4, doodle_id, amount)
        elif doodle_kind == "star":
            for spoke_index in range(3):
                angle = spoke_index * math.pi / 3
                spoke_x, spoke_y = math.cos(angle) * doodle_size, math.sin(angle) * doodle_size
                sketch_line(draw, (x - spoke_x, y - spoke_y), (x + spoke_x, y + spoke_y), color, 4, doodle_id + spoke_index, amount * 3 - spoke_index)
        else:
            squiggle_points = [(x - doodle_size * 1.5 + point_index * doodle_size * 0.5, y + math.sin(point_index * 1.6) * doodle_size * 0.4) for point_index in range(7)]
            if amount > 0:
                draw.line(shake_points(squiggle_points[:2 + int(5 * clamp_zero_to_one(amount))], 2, doodle_id), fill=color, width=4, joint="curve")


# storyboard panel marks over a playing scene: a sketchy border and a scene tag that slides in and back out.
# They stay at the edges so faces and the middle of the picture stay clear
def draw_storyboard_marks(image, part, seconds_into_scene):
    draw = ImageDraw.Draw(image)
    margin = base_size * 0.03
    line_width = max(3, base_size // 300)
    sketch_rectangle(draw, [margin, margin, VIDEO_WIDTH - margin, VIDEO_HEIGHT - margin], "white", line_width, 900 + part["scene_index"], seconds_into_scene / 1.0)
    tag_slide_in = ease_with_overshoot(clamp_zero_to_one((seconds_into_scene - 0.3) / 0.5))
    tag_slide_out = ease_in_out(clamp_zero_to_one((seconds_into_scene - 4.0) / 0.5))
    if tag_slide_out >= 1:
        return
    tag_text = f"SCENE {part['scene_index'] + 1:02d}"
    if part["section_title"]:
        tag_text += "  /  " + part["section_title"].upper()
    padding = label_font.size * 0.5
    tag_width = label_font.getlength(tag_text) + padding * 2
    tag_height = label_font.size * 1.6
    hidden_distance = tag_width + margin * 2
    tag_left = margin * 1.6 - hidden_distance * (1 - tag_slide_in) - hidden_distance * tag_slide_out
    tag_top = margin * 1.6
    draw.rectangle([tag_left, tag_top, tag_left + tag_width, tag_top + tag_height], fill=accent_color)
    sketch_rectangle(draw, [tag_left, tag_top, tag_left + tag_width, tag_top + tag_height], "white", 3, 950 + part["scene_index"])
    draw.text((tag_left + padding, tag_top + label_font.size * 0.25), tag_text, font=label_font, fill=background_color)


# title words slide up one after another and overshoot a little
# kinetic title: each word slams in big and tilted, shrinks to size with an overshoot, then keeps a gentle wobble
def draw_title_word_by_word(image, text, center_y, seconds):
    word_index = 0
    for line_index, line in enumerate(split_into_lines(text, big_title_font, VIDEO_WIDTH * 0.85)):
        x = (VIDEO_WIDTH - big_title_font.getlength(line)) / 2
        y = center_y + line_index * big_title_font.size * 1.2
        for word in line.split():
            word_width = big_title_font.getlength(word)
            word_progress = clamp_zero_to_one((seconds - word_index * 0.12) / 0.45)
            if word_progress > 0:
                landing = ease_with_overshoot(word_progress)
                word_scale = 1.8 - 0.8 * landing
                start_tilt = 15 if word_index % 2 else -15
                word_angle = start_tilt * (1 - landing) + math.sin(seconds * 3 + word_index) * 1.5
                # draw the word on its own see-through image so it can be scaled and rotated
                word_image = Image.new("RGBA", (int(word_width + 40), int(big_title_font.size * 1.5)), (0, 0, 0, 0))
                ImageDraw.Draw(word_image).text((20, 0), word, font=big_title_font, fill=text_color)
                scaled_size = (max(1, int(word_image.width * word_scale)), max(1, int(word_image.height * word_scale)))
                word_image = word_image.resize(scaled_size).rotate(word_angle, expand=True, resample=Image.BICUBIC)
                word_center = (x + word_width / 2, y + big_title_font.size * 0.75)
                image.paste(word_image, (int(word_center[0] - word_image.width / 2), int(word_center[1] - word_image.height / 2)), word_image)
            x += big_title_font.getlength(word + " ")
            word_index += 1


# position and size of one cell in a grid that holds every scene
def get_grid_cell(cell_index):
    columns = math.ceil(math.sqrt(len(scenes)))
    rows = math.ceil(len(scenes) / columns)
    cell_width = VIDEO_WIDTH * 0.9 / columns
    cell_height = cell_width * VIDEO_HEIGHT / VIDEO_WIDTH
    left = VIDEO_WIDTH * 0.05 + (cell_index % columns) * cell_width
    top = (VIDEO_HEIGHT - rows * cell_height) / 2 + (cell_index // columns) * cell_height
    return left + 6, top + 6, cell_width - 12, cell_height - 12


def draw_opener(seconds):
    image = new_background()
    draw = ImageDraw.Draw(image)
    draw_doodles(draw, 11, 10, seconds / 1.5)
    title_top = VIDEO_HEIGHT * 0.16
    draw_title_highlight(image, video_title, title_top, seconds)
    draw_title_word_by_word(image, video_title, title_top, seconds)
    # sketchy underline, then an arrow pointing down at the photos
    underline_y = get_title_bottom(video_title, title_top) + 10
    sketch_line(draw, (VIDEO_WIDTH * 0.3, underline_y), (VIDEO_WIDTH * 0.7, underline_y + 8), text_color, 6, 21, (seconds - 0.8) / 0.4)
    sketch_arrow(draw, (VIDEO_WIDTH * 0.74, underline_y), (VIDEO_WIDTH * 0.68, VIDEO_HEIGHT * 0.52), accent_color, 6, 22, (seconds - 1.3) / 0.5)
    # quick peeks: a few scenes land like photos taped to a storyboard wall
    peek_scenes = scenes[::max(1, len(scenes) // 4)][:4]
    peek_angles = [-4, 3, -2, 5]
    for peek_index, scene in enumerate(peek_scenes):
        progress = clamp_zero_to_one((seconds - 0.8 - peek_index * 0.25) / 0.5)
        if progress > 0:
            center_x = VIDEO_WIDTH * (0.16 + peek_index * 0.227)
            center_y = VIDEO_HEIGHT * 0.7 + (1 - ease_with_overshoot(progress)) * VIDEO_HEIGHT * 0.5
            # photos spin into place as they land
            spin = (1 - ease_with_overshoot(progress)) * (40 if peek_index % 2 else -40)
            paste_taped_photo(image, scene["engraved_cutout"], (center_x, center_y), int(VIDEO_WIDTH * 0.19), peek_angles[peek_index] + spin)
    # the camera slowly pushes in and tilts across the whole opener
    opener_progress = ease_in_out(clamp_zero_to_one(seconds / OPENER_SECONDS))
    return camera_move(image, 1.0 + 0.08 * opener_progress, 1.5 * opener_progress)


def draw_section_card(section_number, title, seconds):
    image = new_background()
    draw = ImageDraw.Draw(image)
    draw_doodles(draw, 100 + section_number, 8, seconds / 1.0)
    number_text = f"{section_number:02d}"
    number_drop = (1 - ease_with_overshoot(clamp_zero_to_one(seconds / 0.5))) * -100
    number_top = VIDEO_HEIGHT * 0.18 + number_drop
    # hatched block offset behind the number, like a pencil shadow
    number_width = number_font.getlength(number_text)
    hatch_left = VIDEO_WIDTH / 2 - number_width * 0.4
    draw_hatched_box(image, [hatch_left, VIDEO_HEIGHT * 0.24, hatch_left + number_width, VIDEO_HEIGHT * 0.24 + number_font.size * 0.9], accent_color, seconds / 0.5)
    # the coming scene as a big engraved cutout, sliding in at the bottom right and drifting slowly
    coming_scene = next(scene for scene in scenes if scene["title"] == title)
    photo_slide = ease_with_overshoot(clamp_zero_to_one(seconds / 0.5))
    photo_center = (VIDEO_WIDTH * (0.82 + 0.4 * (1 - photo_slide) - 0.03 * seconds), VIDEO_HEIGHT * 0.84)
    paste_cutout(image, coming_scene["engraved_cutout"], photo_center, 6 - 2 * seconds, VIDEO_WIDTH * 0.36)
    draw.text(((VIDEO_WIDTH - number_font.getlength(number_text)) / 2, number_top), number_text, font=number_font, fill=text_color)
    # a quick pen circle around the number
    number_center = (VIDEO_WIDTH / 2, number_top + number_font.size * 0.6)
    sketch_circle(draw, number_center, number_font.getlength(number_text) * 0.85, number_font.size * 0.75, accent_color, 7, 200 + section_number, (seconds - 0.2) / 0.6)
    draw_title_highlight(image, title, VIDEO_HEIGHT * 0.52, seconds - 0.2)
    draw_title_word_by_word(image, title, VIDEO_HEIGHT * 0.52, seconds - 0.2)
    # an arrow sweeping in from the corner toward the title
    sketch_arrow(draw, (VIDEO_WIDTH * 0.08, VIDEO_HEIGHT * 0.85), (VIDEO_WIDTH * 0.2, VIDEO_HEIGHT * 0.7), text_color, 5, 300 + section_number, (seconds - 0.6) / 0.5)
    # the camera starts close and tilted, then settles back while still drifting
    card_progress = ease_in_out(clamp_zero_to_one(seconds / SECTION_CARD_SECONDS))
    return camera_move(image, 1.14 - 0.12 * card_progress, -3 * (1 - card_progress) + 0.5 * card_progress)


# closer collage: every clip as an engraved cutout, scattered and tilted around its grid cell. Made once
collage_placer = random.Random(5)
closer_photos = []
for scene_index, scene in enumerate(scenes):
    left, top, width, height = get_grid_cell(scene_index)
    photo_width = int(width * 1.2)
    closer_photos.append({
        "center": (left + width / 2 + collage_placer.uniform(-0.15, 0.15) * width, top + height / 2 + collage_placer.uniform(-0.15, 0.15) * height),
        "angle": collage_placer.uniform(-8, 8),
        "cutout": scene["engraved_cutout"].resize((photo_width, int(photo_width * scene["engraved_cutout"].height / scene["engraved_cutout"].width))),
    })


def draw_closer(seconds):
    collage_image = new_background()
    for scene_index, photo in enumerate(closer_photos):
        # each photo flies up from below and lands on the pile, one after another
        progress = clamp_zero_to_one((seconds - scene_index * 0.04) / 0.6)
        if progress > 0:
            center = (photo["center"][0], photo["center"][1] + (1 - ease_with_overshoot(progress)) * VIDEO_HEIGHT)
            paste_cutout(collage_image, photo["cutout"], center, photo["angle"])
    # after the collage has gathered, dim it (it stays visible) and bring the title back on top
    dim_amount = clamp_zero_to_one((seconds - 2.0) / 0.6) * 0.65
    image = Image.blend(collage_image, new_background(), dim_amount)
    if seconds > 2.2:
        draw = ImageDraw.Draw(image)
        draw_doodles(draw, 77, 10, (seconds - 2.2) / 1.0)
        draw_hatched_box(image, [VIDEO_WIDTH * 0.3, VIDEO_HEIGHT * 0.62, VIDEO_WIDTH * 0.7, VIDEO_HEIGHT * 0.66], accent_color, (seconds - 2.6) / 0.5)
        draw_title_highlight(image, video_title, VIDEO_HEIGHT * 0.4, seconds - 2.2)
        draw_title_word_by_word(image, video_title, VIDEO_HEIGHT * 0.4, seconds - 2.2)
    # the camera slowly pulls back and turns across the whole closer
    closer_progress = ease_in_out(clamp_zero_to_one(seconds / CLOSER_SECONDS))
    return camera_move(image, 1.1 - 0.08 * closer_progress, 2 - 3 * closer_progress)


# ---------- transitions (one per theme) ----------
# direction as an x,y step: where the old picture moves to
def get_direction_step(direction):
    return {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}[direction]


def paste_scaled(canvas, picture, scale, offset_x, offset_y, frame_color=None, corner_radius=0):
    width, height = int(VIDEO_WIDTH * scale), int(VIDEO_HEIGHT * scale)
    left = int((VIDEO_WIDTH - width) / 2 + offset_x)
    top = int((VIDEO_HEIGHT - height) / 2 + offset_y)
    if frame_color:
        ImageDraw.Draw(canvas).rectangle([left - 14, top - 14, left + width + 14, top + height + 14], fill=frame_color)
    corner_mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(corner_mask).rounded_rectangle([0, 0, width, height], radius=corner_radius, fill=255)
    canvas.paste(picture.resize((width, height)), (left, top), corner_mask)


# framed screens in a dark room, the camera steps back and slides to the next one
def gallery_transition(old_picture, new_picture, progress, direction, scene_index):
    canvas = new_background()
    step_x, step_y = get_direction_step(direction)
    camera_scale = 1 - 0.3 * math.sin(math.pi * progress)
    travel = ease_in_out(progress)
    spacing_x, spacing_y = VIDEO_WIDTH * 1.1 * camera_scale, VIDEO_HEIGHT * 1.15 * camera_scale
    paste_scaled(canvas, old_picture, camera_scale, step_x * travel * spacing_x, step_y * travel * spacing_y, accent_color)
    paste_scaled(canvas, new_picture, camera_scale, step_x * (travel - 1) * spacing_x, step_y * (travel - 1) * spacing_y, accent_color)
    return canvas


# the top card is dealt off the deck and the card underneath grows into place
def card_deck_transition(old_picture, new_picture, progress, direction, scene_index):
    # soft gradient from the background colour to a bit of the accent colour
    gradient_mask = Image.linear_gradient("L").resize((VIDEO_WIDTH, VIDEO_HEIGHT)).point(lambda value: value // 3)
    canvas = Image.composite(Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), accent_color), new_background(), gradient_mask)
    step_x, step_y = get_direction_step(direction)
    corner_radius = int(60 * math.sin(math.pi * progress))
    paste_scaled(canvas, new_picture, 0.8 + 0.2 * ease_with_overshoot(progress) if progress < 1 else 1, 0, 0, corner_radius=corner_radius)
    travel = ease_in_out(progress)
    paste_scaled(canvas, old_picture, 1 - 0.15 * math.sin(math.pi * progress), step_x * travel * VIDEO_WIDTH * 1.1, step_y * travel * VIDEO_HEIGHT * 1.1, corner_radius=corner_radius)
    return canvas


# two flat colour shapes sweep across and uncover the next picture
def blocks_transition(old_picture, new_picture, progress, direction, scene_index):
    canvas = (old_picture if progress < 0.5 else new_picture).copy()
    draw = ImageDraw.Draw(canvas)
    step_x, step_y = get_direction_step(direction)
    for block_color, delay in [(accent_color, 0.0), (background_color, 0.12)]:
        block_progress = ease_in_out(clamp_zero_to_one((progress - delay) / (1 - 0.12)))
        # the block travels from fully off one side, across the screen, to fully off the other side
        shift = -1 + 2 * block_progress
        left, top = step_x * shift * VIDEO_WIDTH, step_y * shift * VIDEO_HEIGHT
        if 0 < block_progress < 1:
            draw.rectangle([left, top, left + VIDEO_WIDTH, top + VIDEO_HEIGHT], fill=block_color)
    return canvas


# the old scene shrinks back into the grid of all scenes, then the new one zooms forward
def depth_transition(old_picture, new_picture, progress, direction, scene_index):
    canvas = new_background()
    for cell_index, scene in enumerate(scenes):
        left, top, width, height = get_grid_cell(cell_index)
        canvas.paste(scene["engraved"].resize((int(width), int(height))), (int(left), int(top)))
    canvas = Image.blend(new_background(), canvas, 0.6)
    if progress < 0.5:
        picture, cell_index, amount_shrunk = old_picture, max(0, scene_index - 1), ease_in_out(progress * 2)
    else:
        picture, cell_index, amount_shrunk = new_picture, scene_index, ease_in_out((1 - progress) * 2)
    left, top, width, height = get_grid_cell(cell_index)
    current_left, current_top = left * amount_shrunk, top * amount_shrunk
    current_width = VIDEO_WIDTH + (width - VIDEO_WIDTH) * amount_shrunk
    current_height = VIDEO_HEIGHT + (height - VIDEO_HEIGHT) * amount_shrunk
    canvas.paste(picture.resize((int(current_width), int(current_height))), (int(current_left), int(current_top)))
    return canvas


transition_functions = {"gallery": gallery_transition, "card_deck": card_deck_transition, "blocks": blocks_transition, "depth": depth_transition}
theme_transition = transition_functions[theme["transition"]]


# hand drawn speed lines trailing behind the movement, strongest in the middle of the transition
def draw_speed_lines(canvas, progress, direction, scene_index):
    draw = ImageDraw.Draw(canvas)
    step_x, step_y = get_direction_step(direction)
    line_strength = math.sin(math.pi * progress)
    if line_strength < 0.1:
        return
    for line_index, across in enumerate([0.15, 0.3, 0.7, 0.85]):
        if step_x != 0:
            start = (VIDEO_WIDTH * (0.5 - step_x * 0.25), VIDEO_HEIGHT * across)
            end = (start[0] - step_x * VIDEO_WIDTH * 0.25 * line_strength, start[1])
        else:
            start = (VIDEO_WIDTH * across, VIDEO_HEIGHT * (0.5 - step_y * 0.25))
            end = (start[0], start[1] - step_y * VIDEO_HEIGHT * 0.25 * line_strength)
        sketch_line(draw, start, end, accent_color if line_index % 2 else "white", 6, 700 + scene_index * 4 + line_index)


def draw_transition(old_picture, new_picture, progress, direction, scene_index):
    canvas = theme_transition(old_picture, new_picture, progress, direction, scene_index)
    # motion blur along the direction of travel, strongest when moving fastest (the middle)
    speed = math.sin(math.pi * progress)
    blur_length = int(speed * 60)
    if blur_length > 1:
        blur_size = (blur_length, 1) if direction in ["left", "right"] else (1, blur_length)
        canvas = Image.fromarray(cv2.blur(np.array(canvas), blur_size))
    # the camera leans into the move and zooms slightly, like a whip pan
    lean = 3 if direction in ["left", "up"] else -3
    canvas = camera_move(canvas, 1.0 + 0.05 * speed, lean * speed)
    draw_speed_lines(canvas, progress, direction, scene_index)
    return canvas


# ---------- timeline ----------
# every part knows its length, its first picture and its last picture, so transitions can join any two parts
def count_frames(seconds):
    return round(seconds * FRAMES_PER_SECOND)


parts = [{"kind": "opener", "frames": count_frames(OPENER_SECONDS), "scene_index": 0}]
section_number = 0
section_title = None
for scene_index, scene in enumerate(scenes):
    if scene["title"]:
        section_number += 1
        section_title = scene["title"]
        parts.append({"kind": "card", "frames": count_frames(SECTION_CARD_SECONDS), "scene_index": scene_index, "section_number": section_number, "title": scene["title"]})
    parts.append({"kind": "scene", "frames": count_frames(scene["seconds"]), "scene_index": scene_index, "section_title": section_title})
parts.append({"kind": "closer", "frames": count_frames(CLOSER_SECONDS), "scene_index": len(scenes) - 1})


def get_first_picture(part):
    if part["kind"] == "scene":
        # scenes arrive in the full halftone/hatching look, then fade to real colour
        return stylize_frame(unpack_frame(scenes[part["scene_index"]]["first_frame"]), 1)
    if part["kind"] == "card":
        return draw_section_card(part["section_number"], part["title"], 0)
    return draw_closer(0)


def get_last_picture(part):
    if part["kind"] == "scene":
        return unpack_frame(scenes[part["scene_index"]]["last_frame"])
    if part["kind"] == "card":
        return draw_section_card(part["section_number"], part["title"], SECTION_CARD_SECONDS)
    return draw_opener(OPENER_SECONDS)


# put a transition between each pair of parts. Narration never overlaps because transitions only use frozen frames
timeline = [parts[0]]
for transition_index, (part_before, part_after) in enumerate(zip(parts, parts[1:])):
    direction = SLIDE_DIRECTIONS[transition_index % len(SLIDE_DIRECTIONS)]
    timeline.append({"kind": "transition", "frames": count_frames(TRANSITION_SECONDS), "before": part_before, "after": part_after, "direction": direction, "scene_index": part_after["scene_index"]})
    timeline.append(part_after)

next_start_frame = 0
for part in timeline:
    part["start_seconds"] = next_start_frame / FRAMES_PER_SECOND
    next_start_frame += part["frames"]
total_frames = next_start_frame
total_seconds = total_frames / FRAMES_PER_SECOND


# ---------- preview ----------
if arguments.preview:
    for old_preview_name in os.listdir(preview_folder):
        os.remove(os.path.join(preview_folder, old_preview_name))
    draw_opener(OPENER_SECONDS).save(os.path.join(preview_folder, "000_opener.png"))
    for part_index, part in enumerate(timeline):
        if part["kind"] == "card":
            get_last_picture(part).save(os.path.join(preview_folder, f"{part_index:03d}_card.png"))
        if part["kind"] == "transition":
            old_picture = get_last_picture(part["before"])
            if part["before"]["kind"] == "scene":
                old_picture = stylize_frame(old_picture, ease_in_out(0.7))
            still = draw_transition(old_picture, get_first_picture(part["after"]), 0.35, part["direction"], part["scene_index"])
            still.save(os.path.join(preview_folder, f"{part_index:03d}_transition.png"))
    draw_closer(CLOSER_SECONDS).save(os.path.join(preview_folder, "999_closer.png"))
    # one scene with its storyboard marks, 1.5 seconds in
    first_scene_part = next(part for part in timeline if part["kind"] == "scene")
    scene_still = unpack_frame(scenes[first_scene_part["scene_index"]]["first_frame"])
    scene_still = stylize_frame(scene_still, 0.4)
    draw_storyboard_marks(scene_still, first_scene_part, 1.5)
    draw_closer(1.9).save(os.path.join(preview_folder, "998_closer_collage.png"))
    scene_still.save(os.path.join(preview_folder, "001_scene_marks.png"))
    print("Preview stills saved in", preview_folder)
    print("Video length will be about", round(total_seconds), "seconds")
    raise SystemExit


# ---------- captions ----------
# the narration script for each scene number, used for correct spelling
script_by_scene_number = {}
if os.path.exists(narrative_path):
    narrative_text = open(narrative_path).read()
    for number_text, script_text in re.findall(r"Scene #(\d+).*?Script:\s*(.*?)\n", narrative_text, re.DOTALL):
        script_by_scene_number[int(number_text)] = script_text


def get_heard_words(whisper_model, scene):
    # saved to a file so a second render does not have to transcribe again
    words_path = os.path.join(build_folder, f"words_scene_{scene['number']}.json")
    if os.path.exists(words_path):
        return json.load(open(words_path))
    segments, _ = whisper_model.transcribe(scene["path"], word_timestamps=True)
    heard_words = [{"text": word.word.strip(), "start": word.start, "end": word.end} for segment in segments for word in segment.words]
    json.dump(heard_words, open(words_path, "w"))
    return heard_words


def simplify_word(word):
    return re.sub(r"[^a-z0-9]", "", word.lower())


# keep the script's spelling but whisper's timing
def match_script_to_timing(script_text, heard_words):
    script_words = script_text.split()
    matcher = difflib.SequenceMatcher(None, [simplify_word(word["text"]) for word in heard_words], [simplify_word(word) for word in script_words], autojunk=False)
    timed_words = []
    for _, heard_from, heard_to, script_from, script_to in matcher.get_opcodes():
        if script_from == script_to:
            continue
        start = heard_words[heard_from]["start"] if heard_from < len(heard_words) else heard_words[-1]["end"]
        end = heard_words[heard_to - 1]["end"] if heard_to > heard_from else start
        # share the time evenly between the script words in this stretch
        seconds_per_word = (end - start) / (script_to - script_from)
        for position, word in enumerate(script_words[script_from:script_to]):
            timed_words.append({"text": word, "start": start + position * seconds_per_word, "end": start + (position + 1) * seconds_per_word})
    return timed_words


# 3 to 6 words per caption, ending early at the end of a sentence
def group_into_captions(timed_words):
    captions = []
    current_words = []
    for word in timed_words:
        current_words.append(word)
        sentence_ended = word["text"].endswith((".", "!", "?"))
        if len(current_words) == 5 or (sentence_ended and len(current_words) >= 3):
            captions.append(current_words)
            current_words = []
    if current_words:
        captions.append(current_words)
    return captions


print("Getting word timestamps...")
from faster_whisper import WhisperModel
whisper_model = WhisperModel(WHISPER_MODEL_NAME, device="cpu", compute_type="int8")
for scene in scenes:
    print("  scene", scene["number"])
    timed_words = get_heard_words(whisper_model, scene)
    if scene["number"] in script_by_scene_number and timed_words:
        timed_words = match_script_to_timing(script_by_scene_number[scene["number"]], timed_words)
    scene["captions"] = group_into_captions(timed_words)
del whisper_model


def format_srt_time(seconds):
    milliseconds = round(seconds * 1000)
    return f"{milliseconds // 3600000:02d}:{milliseconds // 60000 % 60:02d}:{milliseconds // 1000 % 60:02d},{milliseconds % 1000:03d}"


srt_blocks = []
for part in timeline:
    if part["kind"] == "scene":
        for caption in scenes[part["scene_index"]]["captions"]:
            start = format_srt_time(part["start_seconds"] + caption[0]["start"])
            end = format_srt_time(part["start_seconds"] + caption[-1]["end"])
            text = " ".join(word["text"] for word in caption)
            srt_blocks.append(f"{len(srt_blocks) + 1}\n{start} --> {end}\n{text}\n")
open(os.path.join(videos_folder, "final_video.srt"), "w").write("\n".join(srt_blocks))


# draw the caption for this moment, with the spoken word in the accent colour
def draw_caption(image, scene, seconds_into_scene):
    for caption in scene["captions"]:
        if caption[0]["start"] <= seconds_into_scene <= caption[-1]["end"] + 0.2:
            break
    else:
        return
    draw = ImageDraw.Draw(image)
    lines = split_into_lines(" ".join(word["text"] for word in caption), caption_font, VIDEO_WIDTH * 0.84)
    # 9:16 captions sit higher, above the app buttons
    bottom_y = VIDEO_HEIGHT * (0.84 if SHAPE == "16:9" else 0.70)
    y = VIDEO_HEIGHT * 0.07 if scene["number"] in CAPTIONS_ON_TOP_SCENES else bottom_y
    word_index = 0
    for line in lines:
        x = (VIDEO_WIDTH - caption_font.getlength(line)) / 2
        for word_text in line.split():
            word = caption[word_index]
            is_spoken_now = seconds_into_scene >= word["start"] and (word_index == len(caption) - 1 or seconds_into_scene < caption[word_index + 1]["start"])
            draw.text((x, y), word_text, font=caption_font, fill=accent_color if is_spoken_now else "white", stroke_width=4, stroke_fill="black")
            x += caption_font.getlength(word_text + " ")
            word_index += 1
        y += caption_font.size * 1.25


# ---------- sound ----------
# reads any audio or video file as a mono array of samples
def load_audio(file_path, audio_filter="anull"):
    command = ["ffmpeg", "-loglevel", "error", "-i", file_path, "-vn", "-af", audio_filter, "-ar", str(AUDIO_SAMPLE_RATE), "-ac", "1", "-f", "f32le", "-"]
    return np.frombuffer(subprocess.check_output(command), dtype=np.float32)


# adds a sound into the big track, starting at the given second
def add_sound(track, sound, start_seconds, volume=1.0):
    start_sample = int(start_seconds * AUDIO_SAMPLE_RATE)
    length = max(0, min(len(sound), len(track) - start_sample))
    track[start_sample:start_sample + length] += sound[:length] * volume


def make_time_array(seconds):
    return np.arange(int(seconds * AUDIO_SAMPLE_RATE)) / AUDIO_SAMPLE_RATE


def smooth_noise(seconds, window_size):
    noise = np.random.uniform(-1, 1, len(make_time_array(seconds)))
    return np.convolve(noise, np.ones(window_size) / window_size, mode="same")


def make_whoosh():
    noise = smooth_noise(TRANSITION_SECONDS, 20)
    return noise * np.sin(np.linspace(0, np.pi, len(noise))) ** 2 * 3


def make_soft_hit():
    time = make_time_array(0.6)
    return np.sin(2 * np.pi * 70 * time) * np.exp(-time * 7)


def make_tick():
    time = make_time_array(0.05)
    return np.sin(2 * np.pi * 1800 * time) * np.exp(-time * 120) * 0.5


def make_riser():
    time = make_time_array(1.5)
    pitch_sweep = np.sin(2 * np.pi * (200 * time + 200 * time ** 2))
    return (pitch_sweep * 0.4 + smooth_noise(1.5, 8)) * (time / 1.5) ** 2


def get_sound_effect(effect_name, make_effect):
    if SFX_FOLDER:
        return load_audio(os.path.join(SFX_FOLDER, effect_name + ".wav"))
    return make_effect()


# repeats the theme's ostinato (a list of note frequencies in Hz) for the whole video
def make_ostinato_music(note_frequencies):
    note_seconds = 0.3
    time = make_time_array(note_seconds)
    # fade each note in quickly and let it die away so there are no clicks
    note_shape = np.minimum(1, time / 0.01) * np.exp(-time * 6) * np.minimum(1, (note_seconds - time) / 0.02)
    notes = []
    for frequency in note_frequencies:
        # the octave above is added so the low notes can be heard on small speakers
        tone = np.sin(2 * np.pi * frequency * time) + 0.5 * np.sin(2 * np.pi * frequency * 2 * time) + 0.25 * np.sin(2 * np.pi * frequency * 4 * time)
        notes.append(tone / 1.75 * note_shape)
    return np.concatenate(notes)


print("Mixing sound...")
total_samples = int(total_seconds * AUDIO_SAMPLE_RATE) + AUDIO_SAMPLE_RATE
voice_track = np.zeros(total_samples, dtype=np.float32)
effects_track = np.zeros(total_samples, dtype=np.float32)

whoosh_sound = get_sound_effect("whoosh", make_whoosh)
hit_sound = get_sound_effect("hit", make_soft_hit)
tick_sound = get_sound_effect("tick", make_tick)
riser_sound = get_sound_effect("riser", make_riser)

for part in timeline:
    start_seconds = part["start_seconds"]
    if part["kind"] == "scene":
        # loudnorm evens out the narration loudness between scenes
        add_sound(voice_track, load_audio(scenes[part["scene_index"]]["path"], "loudnorm=I=-16:TP=-1.5:LRA=11"), start_seconds)
    if part["kind"] == "transition":
        add_sound(effects_track, whoosh_sound, start_seconds, SFX_VOLUME)
        if part["after"]["kind"] == "closer":
            add_sound(effects_track, riser_sound, start_seconds + TRANSITION_SECONDS - len(riser_sound) / AUDIO_SAMPLE_RATE, SFX_VOLUME)
    if part["kind"] == "card":
        add_sound(effects_track, hit_sound, start_seconds, SFX_VOLUME * 2)
        for word_index in range(len(part["title"].split())):
            add_sound(effects_track, tick_sound, start_seconds + 0.2 + word_index * 0.18, SFX_VOLUME)
    if part["kind"] == "opener":
        for word_index in range(len(video_title.split())):
            add_sound(effects_track, tick_sound, start_seconds + word_index * 0.18, SFX_VOLUME)

music_loop = load_audio(MUSIC_FILE) if MUSIC_FILE else make_ostinato_music(theme["music"]["ostinato"])
music_track = np.resize(music_loop, total_samples).astype(np.float32)

# duck the music while someone is talking: measure the voice loudness in half second windows
window_size = AUDIO_SAMPLE_RATE // 2
window_count = total_samples // window_size
window_loudness = np.abs(voice_track[:window_count * window_size]).reshape(window_count, window_size).mean(axis=1)
window_volumes = np.where(window_loudness > 0.01, MUSIC_VOLUME_UNDER_VOICE, MUSIC_VOLUME)
# interpolating between windows makes the volume change smoothly
window_centers = np.arange(window_count) * window_size + window_size // 2
music_track *= np.interp(np.arange(total_samples), window_centers, window_volumes)
# fade the music out at the very end
fade_samples = AUDIO_SAMPLE_RATE * 2
end_sample = int(total_seconds * AUDIO_SAMPLE_RATE)
music_track[end_sample - fade_samples:end_sample] *= np.linspace(1, 0, fade_samples)

final_audio = np.clip(voice_track + effects_track + music_track, -1, 1)[:end_sample]
audio_path = os.path.join(build_folder, "mixed_audio.wav")
with wave.open(audio_path, "wb") as wave_file:
    wave_file.setnchannels(1)
    wave_file.setsampwidth(2)
    wave_file.setframerate(AUDIO_SAMPLE_RATE)
    wave_file.writeframes((final_audio * 32767).astype(np.int16).tobytes())


# ---------- render ----------
# gives every frame of one part of the timeline, one at a time
def make_part_frames(part):
    if part["kind"] == "scene":
        scene = scenes[part["scene_index"]]
        clip_frames = read_clip_frames(scene["path"])
        frame = unpack_frame(scene["first_frame"])
        for frame_index in range(part["frames"]):
            # if the clip runs out a frame early, hold the last one
            frame = next(clip_frames, frame)
            shown_frame = move_scene_camera(frame, frame_index / max(1, part["frames"] - 1), part["scene_index"])
            # print look fades out over the first 2 seconds, back to real colour
            shown_frame = stylize_frame(shown_frame, 1 - ease_in_out(clamp_zero_to_one(frame_index / FRAMES_PER_SECOND / 2.0)))
            draw_storyboard_marks(shown_frame, part, frame_index / FRAMES_PER_SECOND)
            draw_caption(shown_frame, scene, frame_index / FRAMES_PER_SECOND)
            yield shown_frame
        for _ in clip_frames:
            pass
    elif part["kind"] == "transition":
        real_old_picture = get_last_picture(part["before"])
        # a scene that is leaving turns into the print look during the first half of the transition
        styled_old_picture = make_halftone_hatching(real_old_picture) if part["before"]["kind"] == "scene" else real_old_picture
        new_picture = get_first_picture(part["after"])
        for frame_index in range(part["frames"]):
            progress = frame_index / (part["frames"] - 1)
            old_picture = Image.blend(real_old_picture, styled_old_picture, ease_in_out(clamp_zero_to_one(progress * 2)))
            yield draw_transition(old_picture, new_picture, progress, part["direction"], part["scene_index"])
    else:
        for frame_index in range(part["frames"]):
            seconds = frame_index / FRAMES_PER_SECOND
            if part["kind"] == "opener":
                yield draw_opener(seconds)
            elif part["kind"] == "card":
                yield draw_section_card(part["section_number"], part["title"], seconds)
            else:
                yield draw_closer(seconds)


final_video_path = os.path.join(videos_folder, "final_video.mp4")
encode_command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}", "-r", str(FRAMES_PER_SECOND), "-i", "-", "-i", audio_path, "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", final_video_path]
encoder = subprocess.Popen(encode_command, stdin=subprocess.PIPE)

frames_written = 0
progress_bar_height = max(6, VIDEO_HEIGHT // 180)
# a few frames of film grain, cycled so the whole video has a living, printed texture
grain_frames = [np.random.normal(0, 9, (VIDEO_HEIGHT, VIDEO_WIDTH, 1)).astype(np.int16) for _ in range(6)]
for part_index, part in enumerate(timeline):
    print("Rendering part", part_index + 1, "of", len(timeline), part["kind"])
    for frame in make_part_frames(part):
        # the next frame's drawn lines get a new shake every 4 frames
        boil_step = frames_written // 4
        grainy_frame = np.array(frame, dtype=np.int16) + grain_frames[frames_written % len(grain_frames)]
        frame = Image.fromarray(np.clip(grainy_frame, 0, 255).astype(np.uint8))
        # thin progress bar along the very bottom
        bar_width = VIDEO_WIDTH * frames_written / total_frames
        ImageDraw.Draw(frame).rectangle([0, VIDEO_HEIGHT - progress_bar_height, bar_width, VIDEO_HEIGHT], fill=accent_color)
        encoder.stdin.write(frame.tobytes())
        frames_written += 1

encoder.stdin.close()
encoder.wait()

print("Done! Theme", theme_name, "->", final_video_path)
