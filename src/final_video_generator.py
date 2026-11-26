# TO RUN (from the documentary-content folder):
#   preview stills: python src/final_video_generator.py <doc_name> --preview
#   full render:    python src/final_video_generator.py <doc_name>
#   pick a theme:   add --theme desk_collage (or pop_comic, riso_poster, night_archive, zine). Themes live in themes/
#   pick fonts:     add --title-font MyFont.ttf --caption-font MyFont.ttf (files in the fonts/ folder)
# Turns videos/<doc_name>/animated_narrated_clips/scene_1.mp4, scene_2.mp4, ... into one video with a
# screen-printed, cut-paper collage look: a title sequence that tells a tiny story, section cards,
# transitions, sound design, music and captions. The look itself is built in src/video_effects_generator.py
# Outputs videos/<doc_name>/final_video.mp4 and final_video.srt
#
# Optional titles file inputs/<doc_name>/titles.json:
#   {
#     "video_title": "My Video",
#     "scene_titles": { "1": "First Topic", "4": "Next Topic" },
#     "opener_stamp": "1969",
#     "scene_stamps": { "4": "SOLD OUT" },
#     "closer_stamp": "THE END"
#   }
# A scene with a title starts a new topic and gets a section card before it.
# Stamps are optional: the opener, any section card (keyed by the scene number that starts it) and the closer.
import argparse
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
from PIL import Image, ImageDraw, ImageFont

import video_effects_generator
from video_effects_generator import CUT_PADDING, make_cut_piece

# ============ SETTINGS ============
SHAPE = "16:9"                  # "16:9" or "9:16"
FRAMES_PER_SECOND = 30
AUDIO_SAMPLE_RATE = 44100
SCENE_ORDER = []                # example [1, 3, 2]. Empty means number order
CAPTIONS_ON_TOP_SCENES = []     # scene numbers that already have text at the bottom
NIGHT_SCENES = []               # scene numbers that are night or "secret": they print on a dark ink ground
SFX_FOLDER = ""                 # folder with whoosh.wav, hit.wav, tick.wav, riser.wav. Empty means make them with numpy
MUSIC_FILE = ""                 # your own music track. Empty means play the theme's ostinato
WHISPER_MODEL_NAME = "small"
TITLE_FONT = ""                 # a font file name in the fonts/ folder (e.g. "BebasNeue.ttf") or a full path. Empty uses the theme's font
CAPTION_FONT = ""               # same as TITLE_FONT, for captions
ANIMATE_ON_TWOS = True          # graphics hold each picture for 2 frames, like real cut-out animation
OPENER_SECONDS = 7.5            # the title shows at 3 seconds, so this leaves time to read it
SECTION_CARD_SECONDS = 3.5
TRANSITION_SECONDS = 0.8
CLOSER_SECONDS = 7.0
MUSIC_VOLUME = 0.25
MUSIC_VOLUME_UNDER_VOICE = 0.1
SFX_VOLUME = 0.25
JINGLE_VOLUME = 0.7              # the closing jingle is louder than the other sounds
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
    elif os.path.exists(theme_choice_path) and not arguments.preview and open(theme_choice_path).read() in all_theme_names:
        theme_name = open(theme_choice_path).read()
    else:
        theme_name = random.choice(all_theme_names)
    open(theme_choice_path, "w").write(theme_name)
    return theme_name


theme_name = choose_theme_name()
theme = json.load(open(os.path.join(project_folder, "themes", theme_name + ".json")))
HALFTONE_STEP = theme["halftone_step"]      # 20 to 26 px
SCREEN_ANGLES = theme["screen_angles"]      # a different screen angle for each ink layer
print("Theme:", theme_name)


# ---------- palette ----------
# --preview makes a new random palette every time. The full render reuses the previewed palette.
def choose_palette():
    palette_path = os.path.join(build_folder, "palette.json")
    if os.path.exists(palette_path) and not arguments.preview:
        return json.load(open(palette_path))
    palette = video_effects_generator.make_random_palette()
    json.dump(palette, open(palette_path, "w"))
    return palette


palette = choose_palette()
paper_color = palette["paper"]          # the ground
ink_color = palette["ink"]              # every outline, body text, the caption bar and the fade-out
shadow_color = palette["shadow"]        # overprint shadows and misregistration
highlight_color = palette["highlight"]  # keywords and numerals
spot_color = palette["spot"]            # an extra flat fill
alarm_color = palette["alarm"]          # warm red, only for stamps
# the lighter dot colour used on a dark ink ground
night_dot_color = video_effects_generator.mix_colors(ink_color, paper_color, 0.4)
# names the transitions use
background_color, text_color, accent_color = paper_color, ink_color, highlight_color
print("Palette:", palette)


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
giant_number_font = load_font("title", int(base_size * 1.25))   # giant numerals bleed off the frame
caption_font = load_font("caption", int(base_size * 0.038))
label_font = load_font("title", int(base_size * 0.035))
stamp_font = load_font("title", int(base_size * 0.06))
bubble_font = load_font("title", int(base_size * 0.12))
PROP_SIZE = int(base_size * 0.26)


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
# stamp words come only from this video's titles file, so they are always about this story. A missing stamp is simply not shown
opener_stamp = titles.get("opener_stamp", "")
closer_stamp = titles.get("closer_stamp", "")
scene_stamps = titles.get("scene_stamps", {})

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
    })


def unpack_frame(compressed_frame):
    return Image.fromarray(cv2.imdecode(compressed_frame, cv2.IMREAD_COLOR))


def is_night(scene_index):
    return scenes[scene_index]["number"] in NIGHT_SCENES


# every scene photo is printed once as a halftone print on a cut paper piece, used in the opener, cards and closer.
# Photos use a finer screen than full-frame layers so they stay readable at small sizes
print("Printing scene photos...")
for scene_index, scene in enumerate(scenes):
    # the photo is the scene's still picture from images/<doc_name>/scene_N.png
    scene_image_path = os.path.join(project_folder, "images", doc_name, f"scene_{scene['number']}.png")
    half_frame = Image.open(scene_image_path).convert("RGB").resize((VIDEO_WIDTH // 2, VIDEO_HEIGHT // 2))
    printed_photo = video_effects_generator.print_picture(half_frame, palette, 10, SCREEN_ANGLES, night=is_night(scene_index))
    scene["printed_photo"] = make_cut_piece(np.full((VIDEO_HEIGHT // 2, VIDEO_WIDTH // 2), 255, np.uint8), paper_color, palette, 100 + scene_index, content=np.array(printed_photo))
    scene["printed_small"] = printed_photo.resize((VIDEO_WIDTH // 4, VIDEO_HEIGHT // 4))


# ---------- paper, dust and sprites ----------
paper_layer = video_effects_generator.make_paper_layer(VIDEO_WIDTH, VIDEO_HEIGHT)


# last step for every frame: multiply the paper over everything, then the dust (new positions every 6 frames)
def finish_frame(image, frame_number):
    image = Image.fromarray((np.array(image, dtype=np.float32) * paper_layer).astype(np.uint8))
    video_effects_generator.draw_dust(image, frame_number // 6, paper_color, ink_color)
    return image


# a flat ground: paper, or dark ink for night cards
def new_background(dark=False):
    return Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), ink_color if dark else paper_color)


# sprites are cut once and kept, so a moving piece always has the same edge
sprite_cache = {}
caption_sprites = {}   # emptied after every scene so memory stays low


def get_sprite(key, make_sprite, cache=sprite_cache):
    if key not in cache:
        cache[key] = make_sprite()
    return cache[key]


# pastes a sprite so its middle sits on center, scaled and turned
def paste_sprite(image, sprite, center, angle=0, scale=1.0):
    if scale != 1.0:
        sprite = sprite.resize((max(1, int(sprite.width * scale)), max(1, int(sprite.height * scale))))
    if angle:
        sprite = sprite.rotate(angle, expand=True, resample=Image.BICUBIC)
    image.paste(sprite, (int(center[0] - sprite.width / 2), int(center[1] - sprite.height / 2)), sprite)


def split_into_lines(text, font, max_width):
    lines = [""]
    for word in text.split():
        longer_line = (lines[-1] + " " + word).strip()
        if font.getlength(longer_line) > max_width and lines[-1]:
            lines.append(word)
        else:
            lines[-1] = longer_line
    return lines


# ---------- print look for scene frames ----------
def print_scene_frame(frame, scene_index):
    return video_effects_generator.print_picture(frame, palette, HALFTONE_STEP, SCREEN_ANGLES, night=is_night(scene_index))


# amount 1 shows only the print, 0 only the real frame. In between, a coarse dot screen
# grows or shrinks to swap one for the other (no transparency, just dots)
def dissolve_print(real_frame, printed_frame, amount):
    if amount <= 0:
        return real_frame
    if amount >= 1:
        return printed_frame
    cover = video_effects_generator.halftone_mask(np.full((VIDEO_HEIGHT, VIDEO_WIDTH), amount, np.float32), HALFTONE_STEP, 45)
    return Image.fromarray(np.where(cover[:, :, None], np.array(printed_frame), np.array(real_frame)))


# ---------- pieces used by the opener, cards and closer ----------
# kinetic title: each word slams in big and tilted, shrinks to size with an overshoot, then keeps a small wobble.
# Every word is printed twice: the shadow ink knocked out of register, then the main ink on top
def draw_title_word_by_word(image, text, center_x, center_y, max_width, seconds, word_hex):
    word_index = 0
    for line_index, line in enumerate(split_into_lines(text, big_title_font, max_width)):
        x = center_x - big_title_font.getlength(line) / 2
        y = center_y + line_index * big_title_font.size * 1.2
        for word in line.split():
            word_width = big_title_font.getlength(word)
            word_progress = clamp_zero_to_one((seconds - word_index * 0.12) / 0.45)
            if word_progress > 0:
                landing = ease_with_overshoot(word_progress)
                start_tilt = 15 if word_index % 2 else -15
                word_angle = start_tilt * (1 - landing) + math.sin(seconds * 3 + word_index) * 1.5
                word_image = Image.new("RGBA", (int(word_width + 50), int(big_title_font.size * 1.5)), (0, 0, 0, 0))
                word_draw = ImageDraw.Draw(word_image)
                word_draw.text((26, 5), word, font=big_title_font, fill=shadow_color)
                word_draw.text((20, 0), word, font=big_title_font, fill=word_hex)
                paste_sprite(image, word_image, (x + word_width / 2 + 3, y + big_title_font.size * 0.75), word_angle, 1.8 - 0.8 * landing)
            x += big_title_font.getlength(word + " ")
            word_index += 1


# a cut paper strip behind each title line, sliding in from the left
def draw_title_strips(image, text, center_x, center_y, max_width, seconds, strip_hex):
    for line_index, line in enumerate(split_into_lines(text, big_title_font, max_width)):
        strip_width = int(big_title_font.getlength(line) + big_title_font.size * 0.8)
        strip_height = int(big_title_font.size * 1.2)
        strip = get_sprite(("strip", line, strip_hex), lambda: make_cut_piece(np.full((strip_height, strip_width), 255, np.uint8), strip_hex, palette, 40 + line_index))
        slide = ease_with_overshoot(clamp_zero_to_one((seconds - line_index * 0.1) / 0.4))
        if slide > 0:
            strip_center_x = center_x - (1 - slide) * VIDEO_WIDTH
            paste_sprite(image, strip, (strip_center_x, center_y + line_index * big_title_font.size * 1.2 + big_title_font.size * 0.72), -1 + line_index)


# a stamp that slams down: big, then exactly its size, no bounce
def draw_stamp(image, text, center, angle, seconds, stamp_start):
    slam = clamp_zero_to_one((seconds - stamp_start) / 0.12)
    if not text or slam <= 0:
        return
    stamp = get_sprite(("stamp", text), lambda: video_effects_generator.make_stamp(text, stamp_font, alarm_color, len(text) * 13))
    paste_sprite(image, stamp, center, angle, 1.6 - 0.6 * slam)


# spinning sun rays behind everything. Turned at half size, then blown up to cover the frame
def draw_rays(image, ray_hex, angle, grow=1.0):
    ray_size = int(math.hypot(VIDEO_WIDTH, VIDEO_HEIGHT) * 0.6)
    rays = get_sprite(("rays", ray_hex), lambda: video_effects_generator.make_rays(ray_size, ray_hex, palette, 3))
    turned_rays = rays.rotate(angle, resample=Image.BICUBIC)
    full_size = max(2, int(turned_rays.width * 2 * grow))
    paste_sprite(image, turned_rays.resize((full_size, full_size)), (VIDEO_WIDTH / 2, VIDEO_HEIGHT / 2))


# a prop that lands from start_offset with an overshoot and then sways. "character" is a prop too
def draw_prop(image, prop_name, rest_center, start_offset, seconds, land_start, label_text="!"):
    land = ease_with_overshoot(clamp_zero_to_one((seconds - land_start) / 0.4))
    if land <= 0:
        return
    if prop_name == "character":
        sprite = get_sprite(("character", 0), lambda: video_effects_generator.make_character(int(PROP_SIZE * 1.3), spot_color, palette, 11, 0))
    else:
        sprite = get_sprite(("prop", prop_name, label_text), lambda: video_effects_generator.make_prop(prop_name, PROP_SIZE, palette, 60 + len(prop_name), label_text, bubble_font))
    center = (rest_center[0] + start_offset[0] * (1 - land), rest_center[1] + start_offset[1] * (1 - land))
    sway = math.sin(seconds * 2.5 + len(prop_name)) * 4
    paste_sprite(image, sprite, center, sway)
    if prop_name == "clock":
        # the hands spin round while the clock is on screen
        draw = ImageDraw.Draw(image)
        for hand_length, turns_per_second in [(0.3, 1.5), (0.2, 0.15)]:
            hand_angle = 2 * math.pi * turns_per_second * seconds - math.pi / 2
            hand_end = (center[0] + math.cos(hand_angle) * PROP_SIZE * hand_length, center[1] + math.sin(hand_angle) * PROP_SIZE * hand_length)
            draw.line([center, hand_end], fill=ink_color, width=max(4, PROP_SIZE // 30))


# cut paper confetti falling, tumbling and swaying
def draw_confetti(image, seconds, start_seconds, seed):
    fall_seconds = seconds - start_seconds
    if fall_seconds <= 0:
        return
    pieces = get_sprite(("confetti",), lambda: video_effects_generator.make_confetti(palette, int(base_size * 0.035), 12, 5))
    placer = random.Random(seed)
    for piece_index in range(40):
        start_x = placer.uniform(0, VIDEO_WIDTH)
        start_y = -placer.uniform(0.05, 0.5) * VIDEO_HEIGHT
        fall_speed = placer.uniform(0.35, 0.7) * VIDEO_HEIGHT
        sway = placer.uniform(20, 60)
        spin_speed = placer.uniform(-300, 300)
        position = (start_x + math.sin(fall_seconds * 3 + piece_index) * sway, start_y + fall_seconds * fall_speed)
        paste_sprite(image, pieces[piece_index % len(pieces)], position, fall_seconds * spin_speed)


# position and size of one cell in a grid that holds every scene
def get_grid_cell(cell_index):
    columns = math.ceil(math.sqrt(len(scenes)))
    rows = math.ceil(len(scenes) / columns)
    cell_width = VIDEO_WIDTH * 0.9 / columns
    cell_height = cell_width * VIDEO_HEIGHT / VIDEO_WIDTH
    left = VIDEO_WIDTH * 0.05 + (cell_index % columns) * cell_width
    top = (VIDEO_HEIGHT - rows * cell_height) / 2 + (cell_index // columns) * cell_height
    return left + 6, top + 6, cell_width - 12, cell_height - 12


# ---------- title sequence: one goal, a few locations, one reveal ----------
opener_locations = [scenes[0], scenes[len(scenes) // 3], scenes[2 * len(scenes) // 3]]
opener_location_angles = [-4, 3, -2]


def draw_opener(seconds):
    image = new_background()
    reveal = clamp_zero_to_one((seconds - 2.8) / 0.5)
    # act 3 background: rays spin up behind everything once the reveal starts
    if reveal > 0:
        draw_rays(image, spot_color, seconds * 15, ease_with_overshoot(reveal))
    # act 1 and 2, the locations: printed photos land on a long strip and slide past as the camera travels,
    # then drop away for the reveal
    world_shift = -VIDEO_WIDTH * 0.55 * ease_in_out(clamp_zero_to_one((seconds - 1.2) / 1.8))
    drop = ease_in_out(reveal) * VIDEO_HEIGHT
    for location_index, scene in enumerate(opener_locations):
        land = ease_with_overshoot(clamp_zero_to_one((seconds - location_index * 0.35) / 0.5))
        if land > 0:
            photo = scene["printed_photo"]
            center = (VIDEO_WIDTH * (0.32 + location_index * 0.55) + world_shift, VIDEO_HEIGHT * 0.4 - (1 - land) * VIDEO_HEIGHT * 0.8 + drop)
            paste_sprite(image, photo, center, opener_location_angles[location_index], VIDEO_WIDTH * 0.44 / photo.width)
    # the hero of the story: walks in, keeps walking while the world passes, then runs off for the reveal
    walk_in = ease_in_out(clamp_zero_to_one(seconds / 1.0))
    character_x = VIDEO_WIDTH * (-0.1 + 0.3 * walk_in) + VIDEO_WIDTH * 0.9 * ease_in_out(reveal)
    character_y = VIDEO_HEIGHT * 0.74 - abs(math.sin(seconds * 9)) * VIDEO_HEIGHT * 0.02
    pose = int(seconds * 6) % 2
    character = get_sprite(("character", pose), lambda: video_effects_generator.make_character(int(PROP_SIZE * 1.3), spot_color, palette, 11, pose))
    paste_sprite(image, character, (character_x, character_y), math.sin(seconds * 9) * 3)
    # the goal: a speech bubble with a question mark pops up over the hero during act 1 and 2
    bubble_pop = clamp_zero_to_one((seconds - 0.6) / 0.3) - clamp_zero_to_one((seconds - 2.6) / 0.2)
    if bubble_pop > 0:
        bubble = get_sprite(("prop", "speech_bubble", "?"), lambda: video_effects_generator.make_prop("speech_bubble", PROP_SIZE, palette, 70, "?", bubble_font))
        paste_sprite(image, bubble, (character_x + PROP_SIZE * 0.6, character_y - PROP_SIZE * 0.95), 4, max(0.05, ease_with_overshoot(bubble_pop)))
    # act 3, the reveal: starburst, title, stamp
    if reveal > 0:
        starburst = get_sprite(("prop", "big_starburst"), lambda: video_effects_generator.make_prop("starburst", int(base_size * 0.9), palette, 71))
        paste_sprite(image, starburst, (VIDEO_WIDTH / 2, VIDEO_HEIGHT * 0.42), seconds * 10, max(0.05, ease_with_overshoot(reveal)))
        draw_title_strips(image, video_title, VIDEO_WIDTH / 2, VIDEO_HEIGHT * 0.3, VIDEO_WIDTH * 0.8, seconds - 2.95, paper_color)
        draw_title_word_by_word(image, video_title, VIDEO_WIDTH / 2, VIDEO_HEIGHT * 0.3, VIDEO_WIDTH * 0.8, seconds - 3.0, ink_color)
        draw_stamp(image, opener_stamp,(VIDEO_WIDTH * 0.5, VIDEO_HEIGHT * 0.74), -7, seconds, 3.6)
        if "confetti" in theme["props"]:
            draw_confetti(image, seconds, 3.0, 1)
    # the camera slowly pushes in and turns across the whole opener
    opener_progress = ease_in_out(clamp_zero_to_one(seconds / OPENER_SECONDS))
    return camera_move(image, 1.02 + 0.06 * opener_progress, -1 + 2 * opener_progress)


# ---------- section card ----------
# the stamp for a card, looked up by the number of the scene that starts the section
def get_card_stamp(part):
    return scene_stamps.get(str(scenes[part["scene_index"]]["number"]), "")


def draw_section_card(part, seconds):
    section_number = part["section_number"]
    title = part["title"]
    dark_ground = theme["dark_ground"] or is_night(part["scene_index"])
    ground_hex = ink_color if dark_ground else paper_color
    image = new_background(dark_ground)
    if "rays" in theme["props"]:
        draw_rays(image, night_dot_color if dark_ground else spot_color, seconds * 12)
    # giant halftone numeral bleeding off the left edge: flat highlight with a darker halftone
    # overprinted inside the glyph, and a flat shadow-ink copy knocked out of register behind it
    number_text = f"{section_number:02d}"
    number_slide = ease_with_overshoot(clamp_zero_to_one(seconds / 0.45))
    number_center = (VIDEO_WIDTH * 0.2 - (1 - number_slide) * VIDEO_WIDTH * 0.7, VIDEO_HEIGHT * 0.55)
    numeral_mask = get_sprite(("numeral_mask", number_text), lambda: video_effects_generator.make_text_mask(number_text, giant_number_font))
    numeral_shadow = get_sprite(("numeral_shadow", number_text), lambda: make_cut_piece(numeral_mask, shadow_color, palette, section_number * 7, gap=0, shadow=False))
    numeral = get_sprite(("numeral", number_text, dark_ground), lambda: make_cut_piece(numeral_mask, highlight_color, palette, section_number * 7 + 1, dot_hex=ink_color, dot_coverage=0.5, dot_step=HALFTONE_STEP, dot_angle=SCREEN_ANGLES[1], gap_hex=ground_hex))
    paste_sprite(image, numeral_shadow, (number_center[0] + 16, number_center[1] + 11))
    paste_sprite(image, numeral, number_center)
    # the coming scene as a printed photo, sliding in at the bottom right and drifting
    photo = scenes[part["scene_index"]]["printed_photo"]
    photo_slide = ease_with_overshoot(clamp_zero_to_one((seconds - 0.1) / 0.5))
    photo_center = (VIDEO_WIDTH * (0.8 + 0.5 * (1 - photo_slide) - 0.02 * seconds), VIDEO_HEIGHT * 0.78)
    paste_sprite(image, photo, photo_center, 5 - 2 * seconds, VIDEO_WIDTH * 0.34 / photo.width)
    # two props from the theme, a different pair on every card
    card_props = [prop_name for prop_name in theme["props"] if prop_name not in ("rays", "confetti")]
    draw_prop(image, card_props[section_number % len(card_props)], (VIDEO_WIDTH * 0.9, VIDEO_HEIGHT * 0.17), (0, -VIDEO_HEIGHT * 0.6), seconds, 0.2)
    draw_prop(image, card_props[(section_number + 1) % len(card_props)], (VIDEO_WIDTH * 0.46, VIDEO_HEIGHT * 0.82), (0, VIDEO_HEIGHT * 0.6), seconds, 0.35)
    # the title on cut strips, then the stamp slams under it
    draw_title_strips(image, title, VIDEO_WIDTH * 0.62, VIDEO_HEIGHT * 0.22, VIDEO_WIDTH * 0.6, seconds - 0.15, paper_color if dark_ground else spot_color)
    draw_title_word_by_word(image, title, VIDEO_WIDTH * 0.62, VIDEO_HEIGHT * 0.22, VIDEO_WIDTH * 0.6, seconds - 0.25, ink_color)
    draw_stamp(image, get_card_stamp(part),(VIDEO_WIDTH * 0.64, VIDEO_HEIGHT * 0.52), -6, seconds, 0.7)
    if "confetti" in theme["props"]:
        draw_confetti(image, seconds, 0.4, section_number)
    # the camera starts close and tilted, then settles back while still drifting
    card_progress = ease_in_out(clamp_zero_to_one(seconds / SECTION_CARD_SECONDS))
    return camera_move(image, 1.14 - 0.12 * card_progress, -3 * (1 - card_progress) + 0.5 * card_progress)


# ---------- closer ----------
# every clip as a printed cut piece, scattered and tilted around its grid cell. Made once
collage_placer = random.Random(5)
closer_photos = []
for scene_index, scene in enumerate(scenes):
    left, top, width, height = get_grid_cell(scene_index)
    photo_scale = width * 1.2 / scene["printed_photo"].width
    closer_photos.append({
        "center": (left + width / 2 + collage_placer.uniform(-0.15, 0.15) * width, top + height / 2 + collage_placer.uniform(-0.15, 0.15) * height),
        "angle": collage_placer.uniform(-8, 8),
        "sprite": scene["printed_photo"].resize((int(scene["printed_photo"].width * photo_scale), int(scene["printed_photo"].height * photo_scale))),
    })


def draw_closer(seconds):
    image = new_background()
    for scene_index, photo in enumerate(closer_photos):
        # each photo flies up from below and lands on the pile, one after another
        progress = clamp_zero_to_one((seconds - scene_index * 0.04) / 0.6)
        if progress > 0:
            paste_sprite(image, photo["sprite"], (photo["center"][0], photo["center"][1] + (1 - ease_with_overshoot(progress)) * VIDEO_HEIGHT), photo["angle"])
    # a big paper panel lands over the middle of the pile and the title goes on it
    panel_land = ease_with_overshoot(clamp_zero_to_one((seconds - 2.0) / 0.45))
    if panel_land > 0:
        panel = get_sprite(("closer_panel",), lambda: make_cut_piece(np.full((int(VIDEO_HEIGHT * 0.46), int(VIDEO_WIDTH * 0.7)), 255, np.uint8), paper_color, palette, 21))
        paste_sprite(image, panel, (VIDEO_WIDTH / 2, VIDEO_HEIGHT * 0.45 + (1 - panel_land) * VIDEO_HEIGHT), -1.5)
        draw_title_strips(image, video_title, VIDEO_WIDTH / 2, VIDEO_HEIGHT * 0.3, VIDEO_WIDTH * 0.62, seconds - 2.3, spot_color)
        draw_title_word_by_word(image, video_title, VIDEO_WIDTH / 2, VIDEO_HEIGHT * 0.3, VIDEO_WIDTH * 0.62, seconds - 2.4, ink_color)
        draw_stamp(image, closer_stamp,(VIDEO_WIDTH * 0.66, VIDEO_HEIGHT * 0.6), 8, seconds, 3.1)
    if "confetti" in theme["props"]:
        draw_confetti(image, seconds, 2.4, 99)
    # the camera slowly pulls back and turns across the whole closer
    closer_progress = ease_in_out(clamp_zero_to_one(seconds / CLOSER_SECONDS))
    image = camera_move(image, 1.1 - 0.08 * closer_progress, 2 - 3 * closer_progress)
    # fade out to the dark ink: halftone dots grow until the frame is solid ink
    fade = clamp_zero_to_one((seconds - (CLOSER_SECONDS - 0.9)) / 0.8)
    if fade > 0:
        fade_dots = video_effects_generator.halftone_mask(np.full((VIDEO_HEIGHT, VIDEO_WIDTH), ease_in_out(fade), np.float32), HALFTONE_STEP, 45)
        faded = np.array(image)
        faded[fade_dots] = video_effects_generator.hex_to_rgb(ink_color).astype(np.uint8)
        image = Image.fromarray(faded)
    return image


# ---------- pieces laid over a playing scene ----------
# a paper mat with a hand-cut window, laid over the footage. Its small shadow falls onto the picture
def paste_mat(image):
    margin = int(base_size * 0.02)
    def make_mat():
        mat_mask = np.full((VIDEO_HEIGHT, VIDEO_WIDTH), 255, np.uint8)
        mat_mask[margin:VIDEO_HEIGHT - margin, margin:VIDEO_WIDTH - margin] = 0
        return make_cut_piece(mat_mask, paper_color, palette, 31, gap=0)
    mat = get_sprite(("mat",), make_mat)
    image.paste(mat, (-CUT_PADDING, -CUT_PADDING), mat)


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


# framed prints on a paper wall, the camera steps back and slides to the next one
def gallery_transition(old_picture, new_picture, progress, direction, scene_index):
    canvas = new_background()
    step_x, step_y = get_direction_step(direction)
    camera_scale = 1 - 0.3 * math.sin(math.pi * progress)
    travel = ease_in_out(progress)
    spacing_x, spacing_y = VIDEO_WIDTH * 1.1 * camera_scale, VIDEO_HEIGHT * 1.15 * camera_scale
    paste_scaled(canvas, old_picture, camera_scale, step_x * travel * spacing_x, step_y * travel * spacing_y, ink_color)
    paste_scaled(canvas, new_picture, camera_scale, step_x * (travel - 1) * spacing_x, step_y * (travel - 1) * spacing_y, ink_color)
    return canvas


# the top card is dealt off the deck and the card underneath grows into place, on a flat spot-ink ground
def card_deck_transition(old_picture, new_picture, progress, direction, scene_index):
    canvas = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), spot_color)
    step_x, step_y = get_direction_step(direction)
    corner_radius = int(60 * math.sin(math.pi * progress))
    paste_scaled(canvas, new_picture, 0.8 + 0.2 * ease_with_overshoot(progress) if progress < 1 else 1, 0, 0, corner_radius=corner_radius)
    travel = ease_in_out(progress)
    paste_scaled(canvas, old_picture, 1 - 0.15 * math.sin(math.pi * progress), step_x * travel * VIDEO_WIDTH * 1.1, step_y * travel * VIDEO_HEIGHT * 1.1, corner_radius=corner_radius)
    return canvas


# big hand-cut paper sheets in flat inks sweep across and uncover the next picture
def blocks_transition(old_picture, new_picture, progress, direction, scene_index):
    canvas = (old_picture if progress < 0.5 else new_picture).copy()
    step_x, step_y = get_direction_step(direction)
    for block_color, delay in [(highlight_color, 0.0), (shadow_color, 0.08), (ink_color, 0.16)]:
        block = get_sprite(("block", block_color), lambda: make_cut_piece(np.full((VIDEO_HEIGHT, VIDEO_WIDTH), 255, np.uint8), block_color, palette, 50))
        block_progress = ease_in_out(clamp_zero_to_one((progress - delay) / (1 - 0.16)))
        # the sheet travels from fully off one side, across the screen, to fully off the other side
        shift = -1 + 2 * block_progress
        if 0 < block_progress < 1:
            canvas.paste(block, (int(step_x * shift * VIDEO_WIDTH) - CUT_PADDING, int(step_y * shift * VIDEO_HEIGHT) - CUT_PADDING), block)
    return canvas


# the old scene shrinks back into a grid of all the printed scenes, then the new one zooms forward
def depth_transition(old_picture, new_picture, progress, direction, scene_index):
    canvas = new_background(dark=True)
    for cell_index, scene in enumerate(scenes):
        left, top, width, height = get_grid_cell(cell_index)
        canvas.paste(scene["printed_small"].resize((int(width), int(height))), (int(left), int(top)))
    if progress < 0.5:
        picture, cell_index, amount_shrunk = old_picture, max(0, scene_index - 1), ease_in_out(progress * 2)
    else:
        picture, cell_index, amount_shrunk = new_picture, scene_index, ease_in_out((1 - progress) * 2)
    left, top, width, height = get_grid_cell(cell_index)
    current_width = VIDEO_WIDTH + (width - VIDEO_WIDTH) * amount_shrunk
    current_height = VIDEO_HEIGHT + (height - VIDEO_HEIGHT) * amount_shrunk
    canvas.paste(picture.resize((int(current_width), int(current_height))), (int(left * amount_shrunk), int(top * amount_shrunk)))
    return canvas


transition_functions = {"gallery": gallery_transition, "card_deck": card_deck_transition, "blocks": blocks_transition, "depth": depth_transition}
theme_transition = transition_functions[theme["transition"]]


def draw_transition(old_picture, new_picture, progress, direction, scene_index):
    canvas = theme_transition(old_picture, new_picture, progress, direction, scene_index)
    # the camera leans into the move and zooms slightly, like a whip pan
    speed = math.sin(math.pi * progress)
    lean = 3 if direction in ["left", "up"] else -3
    return camera_move(canvas, 1.0 + 0.05 * speed, lean * speed)


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
        # scenes arrive fully printed, then dissolve to the real footage
        return print_scene_frame(unpack_frame(scenes[part["scene_index"]]["first_frame"]), part["scene_index"])
    if part["kind"] == "card":
        return draw_section_card(part, 0)
    return draw_closer(0)


def get_last_picture(part):
    if part["kind"] == "scene":
        return unpack_frame(scenes[part["scene_index"]]["last_frame"])
    if part["kind"] == "card":
        return draw_section_card(part, SECTION_CARD_SECONDS)
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
def save_still(image, file_name):
    finish_frame(image, 0).save(os.path.join(preview_folder, file_name))


if arguments.preview:
    for old_preview_name in os.listdir(preview_folder):
        os.remove(os.path.join(preview_folder, old_preview_name))
    # the three acts of the title sequence
    save_still(draw_opener(1.0), "000_opener_1_goal.png")
    save_still(draw_opener(2.2), "000_opener_2_locations.png")
    save_still(draw_opener(OPENER_SECONDS - 0.2), "000_opener_3_reveal.png")
    for part_index, part in enumerate(timeline):
        if part["kind"] == "card":
            save_still(draw_section_card(part, 1.0), f"{part_index:03d}_card.png")
        if part["kind"] == "transition":
            old_picture = get_last_picture(part["before"])
            if part["before"]["kind"] == "scene":
                old_picture = dissolve_print(old_picture, print_scene_frame(old_picture, part["before"]["scene_index"]), ease_in_out(0.7))
            save_still(draw_transition(old_picture, get_first_picture(part["after"]), 0.35, part["direction"], part["scene_index"]), f"{part_index:03d}_transition.png")
    # one scene half way through its print dissolve, with the mat and tag
    scene_still = move_scene_camera(unpack_frame(scenes[0]["first_frame"]), 0, 0)
    scene_still = dissolve_print(scene_still, print_scene_frame(scene_still, 0), 0.5)
    paste_mat(scene_still)
    save_still(scene_still, "001_scene_dissolving.png")
    # every prop, the hero, a stamp and a numeral on one sheet
    pieces_sheet = new_background()
    all_prop_names = ["starburst", "speech_bubble", "sticky_note", "folder", "clock", "character"]
    for prop_index, prop_name in enumerate(all_prop_names):
        draw_prop(pieces_sheet, prop_name, (VIDEO_WIDTH * (0.1 + prop_index * 0.16), VIDEO_HEIGHT * 0.25), (0, 0), 1.0, 0)
    draw_stamp(pieces_sheet, opener_stamp or "SAMPLE STAMP",(VIDEO_WIDTH * 0.3, VIDEO_HEIGHT * 0.72), -7, 1.0, 0)
    sample_numeral = make_cut_piece(video_effects_generator.make_text_mask("07", load_font("title", int(base_size * 0.5))), highlight_color, palette, 9, dot_hex=ink_color, dot_coverage=0.5, dot_step=HALFTONE_STEP, dot_angle=SCREEN_ANGLES[1])
    paste_sprite(pieces_sheet, sample_numeral, (VIDEO_WIDTH * 0.72, VIDEO_HEIGHT * 0.7))
    save_still(pieces_sheet, "002_print_pieces.png")
    save_still(draw_closer(1.9), "998_closer_collage.png")
    save_still(draw_closer(CLOSER_SECONDS - 1.0), "999_closer_title.png")
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


# draw the caption for this moment: paper-coloured words on a hand-cut dark ink bar, with a shadow-ink
# copy of the bar knocked out of register behind it. The spoken word is in the highlight ink
def draw_caption(image, scene, seconds_into_scene):
    for caption_index, caption in enumerate(scene["captions"]):
        if caption[0]["start"] <= seconds_into_scene <= caption[-1]["end"] + 0.2:
            break
    else:
        return
    lines = split_into_lines(" ".join(word["text"] for word in caption), caption_font, VIDEO_WIDTH * 0.5)
    line_height = caption_font.size * 1.3
    padding = caption_font.size * 0.6
    bar_width = int(max(caption_font.getlength(line) for line in lines) + padding * 2)
    bar_height = int(len(lines) * line_height + padding * 0.9)
    # captions sit in the left corner, this far in from the edge of the video
    bar_left = int(base_size * 0.045)
    bar_center_x = bar_left + bar_width / 2
    # 9:16 captions sit higher, above the app buttons
    if scene["number"] in CAPTIONS_ON_TOP_SCENES:
        bar_center_y = VIDEO_HEIGHT * 0.1 + bar_height / 2
    else:
        bar_center_y = VIDEO_HEIGHT * (0.95 if SHAPE == "16:9" else 0.75) - bar_height / 2
    bar_mask = np.full((bar_height, bar_width), 255, np.uint8)
    shadow_bar = get_sprite(("shadow", caption_index), lambda: make_cut_piece(bar_mask, shadow_color, palette, caption_index * 2, gap=0, shadow=False), caption_sprites)
    bar = get_sprite(("bar", caption_index), lambda: make_cut_piece(bar_mask, ink_color, palette, caption_index * 2 + 1, gap=0), caption_sprites)
    paste_sprite(image, shadow_bar, (bar_center_x + 9, bar_center_y + 7))
    paste_sprite(image, bar, (bar_center_x, bar_center_y))
    draw = ImageDraw.Draw(image)
    y = bar_center_y - bar_height / 2 + padding * 0.35
    word_index = 0
    for line in lines:
        x = bar_left + padding
        for word_text in line.split():
            word = caption[word_index]
            is_spoken_now = seconds_into_scene >= word["start"] and (word_index == len(caption) - 1 or seconds_into_scene < caption[word_index + 1]["start"])
            draw.text((x, y), word_text, font=caption_font, fill=highlight_color if is_spoken_now else paper_color)
            x += caption_font.getlength(word_text + " ")
            word_index += 1
        y += line_height


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


# a short tune with random notes from a happy major pentatonic scale, different every run.
# It climbs to the top note at the end so it sounds finished
def make_random_jingle():
    jingle_maker = random.Random()
    root_frequency = jingle_maker.choice([392.0, 440.0, 523.25, 587.33])
    scale_steps = [0, 2, 4, 7, 9, 12, 14, 16]
    note_count = jingle_maker.randint(5, 8)
    steps = [jingle_maker.choice(scale_steps) for _ in range(note_count - 1)] + [12]
    notes = []
    for note_index, step in enumerate(steps):
        # the last note rings longer
        note_seconds = 0.6 if note_index == len(steps) - 1 else jingle_maker.choice([0.15, 0.2, 0.3])
        time = make_time_array(note_seconds)
        frequency = root_frequency * 2 ** (step / 12)
        note_shape = np.minimum(1, time / 0.01) * np.exp(-time * 4) * np.minimum(1, (note_seconds - time) / 0.02)
        # a bell-like tone: the note plus a softer octave
        tone = np.sin(2 * np.pi * frequency * time) + 0.4 * np.sin(2 * np.pi * frequency * 2 * time)
        notes.append(tone / 1.4 * note_shape)
    print("Jingle notes:", steps)
    return np.concatenate(notes)


print("Mixing sound...")
total_samples = int(total_seconds * AUDIO_SAMPLE_RATE) + AUDIO_SAMPLE_RATE
voice_track = np.zeros(total_samples, dtype=np.float32)
effects_track = np.zeros(total_samples, dtype=np.float32)

whoosh_sound = get_sound_effect("whoosh", make_whoosh)
hit_sound = get_sound_effect("hit", make_soft_hit)
tick_sound = get_sound_effect("tick", make_tick)
riser_sound = get_sound_effect("riser", make_riser)
jingle_sound = make_random_jingle()

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
            add_sound(effects_track, tick_sound, start_seconds + 0.25 + word_index * 0.12, SFX_VOLUME)
        # the stamp slamming down
        if get_card_stamp(part):
            add_sound(effects_track, hit_sound, start_seconds + 0.7, SFX_VOLUME * 2)
    if part["kind"] == "opener":
        # title words slam in during the reveal, then the stamp
        for word_index in range(len(video_title.split())):
            add_sound(effects_track, tick_sound, start_seconds + 3.0 + word_index * 0.12, SFX_VOLUME)
        if opener_stamp:
            add_sound(effects_track, hit_sound, start_seconds + 3.6, SFX_VOLUME * 2)
    if part["kind"] == "closer":
        # the jingle starts as the photos pile up into the collage
        add_sound(effects_track, jingle_sound, start_seconds + 0.3, JINGLE_VOLUME)
    if part["kind"] == "closer" and closer_stamp:
        add_sound(effects_track, hit_sound, start_seconds + 3.1, SFX_VOLUME * 2)

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
# with ANIMATE_ON_TWOS each graphic picture is held for 2 frames, like real cut-out animation
def on_twos(frame_index):
    return frame_index - frame_index % 2 if ANIMATE_ON_TWOS else frame_index


# gives every frame of one part of the timeline, one at a time
def make_part_frames(part):
    if part["kind"] == "scene":
        scene = scenes[part["scene_index"]]
        caption_sprites.clear()
        clip_frames = read_clip_frames(scene["path"])
        frame = unpack_frame(scene["first_frame"])
        for frame_index in range(part["frames"]):
            # if the clip runs out a frame early, hold the last one
            frame = next(clip_frames, frame)
            shown_frame = move_scene_camera(frame, frame_index / max(1, part["frames"] - 1), part["scene_index"])
            # the print look dissolves away over the first 2 seconds, back to the real footage
            print_amount = 1 - ease_in_out(clamp_zero_to_one(frame_index / FRAMES_PER_SECOND / 2.0))
            if print_amount > 0:
                shown_frame = dissolve_print(shown_frame, print_scene_frame(shown_frame, part["scene_index"]), print_amount)
            paste_mat(shown_frame)
            draw_caption(shown_frame, scene, frame_index / FRAMES_PER_SECOND)
            yield shown_frame
        for _ in clip_frames:
            pass
    elif part["kind"] == "transition":
        real_old_picture = get_last_picture(part["before"])
        leaving_scene = part["before"]["kind"] == "scene"
        # a scene that is leaving turns into the print during the first half of the transition
        printed_old_picture = print_scene_frame(real_old_picture, part["before"]["scene_index"]) if leaving_scene else real_old_picture
        new_picture = get_first_picture(part["after"])
        for frame_index in range(part["frames"]):
            progress = on_twos(frame_index) / (part["frames"] - 1)
            old_picture = dissolve_print(real_old_picture, printed_old_picture, ease_in_out(clamp_zero_to_one(progress * 2))) if leaving_scene else real_old_picture
            yield draw_transition(old_picture, new_picture, progress, part["direction"], part["scene_index"])
    else:
        for frame_index in range(part["frames"]):
            seconds = on_twos(frame_index) / FRAMES_PER_SECOND
            if part["kind"] == "opener":
                yield draw_opener(seconds)
            elif part["kind"] == "card":
                yield draw_section_card(part, seconds)
            else:
                yield draw_closer(seconds)


final_video_path = os.path.join(videos_folder, "final_video.mp4")
encode_command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}", "-r", str(FRAMES_PER_SECOND), "-i", "-", "-i", audio_path, "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", final_video_path]
encoder = subprocess.Popen(encode_command, stdin=subprocess.PIPE)

frames_written = 0
progress_bar_height = max(6, VIDEO_HEIGHT // 180)
for part_index, part in enumerate(timeline):
    print("Rendering part", part_index + 1, "of", len(timeline), part["kind"])
    for frame in make_part_frames(part):
        # thin progress bar along the very bottom in the highlight ink
        bar_width = VIDEO_WIDTH * frames_written / total_frames
        ImageDraw.Draw(frame).rectangle([0, VIDEO_HEIGHT - progress_bar_height, bar_width, VIDEO_HEIGHT], fill=highlight_color)
        # paper multiplied over everything, then dust
        frame = finish_frame(frame, frames_written)
        encoder.stdin.write(frame.tobytes())
        frames_written += 1

encoder.stdin.close()
encoder.wait()

print("Done! Theme", theme_name, "->", final_video_path)
