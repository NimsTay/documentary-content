# The screen-printed, cut-paper look used by final_video_generator.py.
# Everything here only BUILDS images (numpy arrays or PIL images): palettes, the paper layer, dust,
# halftone screens, scissor-cut pieces, stamps, props and characters. The generator decides where they go.
#
# Colour rules: a paper ground, one dark ink and four flat spot inks. Flat inks only:
# tone comes from halftone dot size and from inks overprinting (multiplying) each other.
import colorsys
import math
import random

import cv2
import numpy as np
from PIL import Image, ImageColor, ImageDraw

# every cut piece is built with this much empty space around it (room for the cut gap and shadow)
CUT_PADDING = 32


# ---------- colours ----------
def hex_to_rgb(hex_color):
    return np.array(ImageColor.getrgb(hex_color), dtype=np.float32)


def rgb_to_hex(rgb):
    return "#%02x%02x%02x" % (int(rgb[0]), int(rgb[1]), int(rgb[2]))


def hsv_to_hex(hue, saturation, value):
    red, green, blue = colorsys.hsv_to_rgb(hue % 1, saturation, value)
    return rgb_to_hex((red * 255, green * 255, blue * 255))


# a flat colour part way between two colours (used once to make a new flat ink, never as a gradient)
def mix_colors(first_hex, second_hex, amount):
    return rgb_to_hex(hex_to_rgb(first_hex) * (1 - amount) + hex_to_rgb(second_hex) * amount)


# paper, dark ink (outlines, text, caption bar, fade-out), shadow (overprint/misregistration),
# highlight (keywords, numerals), spot (extra flat fill), alarm (warm red, stamps only)
PRESET_PALETTES = [
    {"paper": "#F2E8D0", "ink": "#1B2550", "shadow": "#2E5BFF", "highlight": "#FF4FA0", "spot": "#FFD400", "alarm": "#E8322B"},  # cream / navy / cobalt / hot pink / yellow / red
    {"paper": "#E9D7AE", "ink": "#2A1B12", "shadow": "#13807C", "highlight": "#E0A526", "spot": "#7A8B3A", "alarm": "#E34A27"},  # manila / sepia-black / teal / mustard / vermilion
    {"paper": "#F5F2E9", "ink": "#1E3B2B", "shadow": "#7EC3EA", "highlight": "#D99A2B", "spot": "#F2A7B5", "alarm": "#F0543C"},  # off-white / forest / sky / ochre / tomato
    {"paper": "#EDEAE2", "ink": "#1A1A1A", "shadow": "#4F7CAC", "highlight": "#F7C948", "spot": "#9BC1A6", "alarm": "#D7263D"},  # newsprint
    {"paper": "#EEF0DC", "ink": "#23304A", "shadow": "#3FA796", "highlight": "#FF8C42", "spot": "#C7B8EA", "alarm": "#E63946"},  # pistachio
    {"paper": "#F7E6DC", "ink": "#2D1E2F", "shadow": "#5B8FB9", "highlight": "#FFC145", "spot": "#6BBF59", "alarm": "#E4572E"},  # blush
    {"paper": "#E6EEF0", "ink": "#0F2A3D", "shadow": "#2A9D8F", "highlight": "#F4A261", "spot": "#E9C46A", "alarm": "#E76F51"},  # sea
    {"paper": "#F3EDE2", "ink": "#2B1A3D", "shadow": "#7B5EA7", "highlight": "#7FD6C2", "spot": "#FFB703", "alarm": "#E5383B"},  # plum
    {"paper": "#F0EAD8", "ink": "#3B2416", "shadow": "#4BA3A3", "highlight": "#F28FAD", "spot": "#B6D369", "alarm": "#D64933"},  # mint and cocoa
    {"paper": "#F4E9DA", "ink": "#202C39", "shadow": "#3E7CB1", "highlight": "#F2C14E", "spot": "#81B29A", "alarm": "#D1495B"},  # harbour
    {"paper": "#ECE4CF", "ink": "#2E2A1E", "shadow": "#6A7D3A", "highlight": "#F18F01", "spot": "#99C1DE", "alarm": "#C73E1D"},  # olive
    {"paper": "#F6F1E7", "ink": "#101B3F", "shadow": "#3D5AFE", "highlight": "#FFE066", "spot": "#FF9F9F", "alarm": "#EF3E36"},  # ultramarine
    {"paper": "#FBF3DC", "ink": "#1C2B2D", "shadow": "#0FA3B1", "highlight": "#F7B32B", "spot": "#F45B69", "alarm": "#E03616"},  # sunshine
    {"paper": "#EFE9F2", "ink": "#241E4E", "shadow": "#8D86C9", "highlight": "#F9C80E", "spot": "#6FD08C", "alarm": "#EA3546"},  # lavender
    {"paper": "#EADFC8", "ink": "#1D1B1A", "shadow": "#2F6690", "highlight": "#D9822B", "spot": "#A7C4A0", "alarm": "#C1292E"},  # rust
    {"paper": "#FFF4E6", "ink": "#143642", "shadow": "#39A0ED", "highlight": "#FF66B3", "spot": "#FFE45E", "alarm": "#F25C54"},  # bubblegum
]


# picks a preset 40% of the time, otherwise builds a brand new palette that still follows the colour rules
def make_random_palette():
    if random.random() < 0.4:
        return dict(random.choice(PRESET_PALETTES))
    paper = hsv_to_hex(random.uniform(0.06, 0.16), random.uniform(0.06, 0.28), random.uniform(0.88, 0.97))
    # dark inks are navy, black-brown, deep green, plum or petrol
    ink_hue = random.choice([0.62, 0.66, 0.07, 0.05, 0.38, 0.45, 0.78, 0.53])
    ink = hsv_to_hex(ink_hue, random.uniform(0.45, 0.8), random.uniform(0.12, 0.24))
    highlight_hue = random.random()
    return {
        "paper": paper,
        "ink": ink,
        "highlight": hsv_to_hex(highlight_hue, random.uniform(0.7, 0.95), random.uniform(0.88, 1.0)),
        "shadow": hsv_to_hex(highlight_hue + random.uniform(0.3, 0.45), random.uniform(0.55, 0.85), random.uniform(0.55, 0.8)),
        "spot": hsv_to_hex(highlight_hue + random.uniform(0.55, 0.75), random.uniform(0.45, 0.75), random.uniform(0.75, 0.95)),
        "alarm": hsv_to_hex(random.uniform(-0.02, 0.03), random.uniform(0.75, 0.9), random.uniform(0.8, 0.92)),
    }


# ---------- paper layer (built once, multiplied over every frame) ----------
def make_paper_layer(width, height):
    noise_maker = np.random.default_rng()
    # mottling: a small random tile blown up so it turns into soft blotches, plus a finer one
    big_tile = noise_maker.uniform(0.93, 1.0, (height // 90 + 2, width // 90 + 2)).astype(np.float32)
    tone = cv2.resize(big_tile, (width, height), interpolation=cv2.INTER_CUBIC)
    small_tile = noise_maker.uniform(0.97, 1.0, (height // 12 + 2, width // 12 + 2)).astype(np.float32)
    tone *= cv2.resize(small_tile, (width, height), interpolation=cv2.INTER_CUBIC)
    # squeegee streaks: a few columns of random values stretched across the width become long faint bands
    streak_tile = noise_maker.uniform(0.975, 1.0, (height // 3, 5)).astype(np.float32)
    tone *= cv2.resize(streak_tile, (width, height), interpolation=cv2.INTER_CUBIC)
    rows, columns = np.mgrid[0:height, 0:width].astype(np.float32)
    # radial vignette
    distance_from_center = ((rows / height - 0.5) ** 2 + (columns / width - 0.5) ** 2) * 2
    tone *= 1 - 0.16 * distance_from_center
    # faint centre fold: a thin dark crease with a soft shade on one side
    fold_distance = columns - width / 2
    tone *= 1 - 0.05 * np.exp(-fold_distance ** 2 / 8) - 0.015 * np.exp(-fold_distance ** 2 / 2000) * (fold_distance > 0)
    # faint fibres: short light hairs in the paper
    fibre_image = Image.new("L", (width, height), 255)
    fibre_draw = ImageDraw.Draw(fibre_image)
    for _ in range(400):
        x, y = noise_maker.uniform(0, width), noise_maker.uniform(0, height)
        angle = noise_maker.uniform(0, math.pi)
        length = noise_maker.uniform(6, 24)
        fibre_draw.line([(x, y), (x + math.cos(angle) * length, y + math.sin(angle) * length)], fill=int(noise_maker.uniform(225, 240)), width=1)
    tone *= np.array(fibre_image, dtype=np.float32) / 255
    # per-pixel noise with a slight warm bias: blue is held back a little more than red
    pixel_noise = noise_maker.normal(0, 0.012, (height, width)).astype(np.float32)
    layer = np.stack([tone + pixel_noise, tone + pixel_noise * 0.95 - 0.004, tone + pixel_noise * 0.9 - 0.014], axis=2)
    return np.clip(layer, 0, 1).astype(np.float32)


# ---------- dust (drawn on every frame, new positions every 6 frames) ----------
# paper-coloured specks (ink voids, they only show on dark ink) and fewer ink specks
def draw_dust(image, seed, paper_hex, ink_hex):
    speck_placer = random.Random(seed)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for speck_color, speck_count, biggest in [(paper_hex, 150, 1.5), (ink_hex, 30, 1.2)]:
        for _ in range(speck_count):
            x, y = speck_placer.uniform(0, width), speck_placer.uniform(0, height)
            radius = speck_placer.uniform(0.5, biggest)
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=speck_color)


# ---------- halftone ----------
# for every pixel: how far it is from the middle of its dot, on a grid turned to the screen angle. Made once per size
screen_distance_cache = {}


def get_screen_distance(width, height, step, angle_degrees):
    cache_key = (width, height, step, angle_degrees)
    if cache_key not in screen_distance_cache:
        rows, columns = np.mgrid[0:height, 0:width].astype(np.float32)
        angle = math.radians(angle_degrees)
        along = columns * math.cos(angle) + rows * math.sin(angle)
        across = -columns * math.sin(angle) + rows * math.cos(angle)
        along -= np.round(along / step) * step
        across -= np.round(across / step) * step
        screen_distance_cache[cache_key] = np.hypot(along, across) / step
    return screen_distance_cache[cache_key]


# coverage is an array of 0 to 1 ink amounts. Returns True where a dot is. Bigger coverage means bigger dots,
# and at 1 the dots join into solid ink. Same result as drawing every circle of the rotated grid as one path
def halftone_mask(coverage, step, angle_degrees):
    height, width = coverage.shape
    smooth_coverage = cv2.blur(coverage.astype(np.float32), (step, step))
    return get_screen_distance(width, height, step, angle_degrees) < 0.72 * np.sqrt(np.clip(smooth_coverage, 0, 1))


# turns a picture into a print: three flat inks as halftone screens at different angles,
# overprinting each other, with the shadow layer knocked a few pixels out of register
def print_picture(picture, palette, step, screen_angles, night=False):
    darkness = 1 - np.array(picture.convert("L"), dtype=np.float32) / 255
    darkness = np.clip((darkness - 0.08) * 1.25, 0, 1)
    height, width = darkness.shape
    printed = np.empty((height, width, 3), dtype=np.float32)
    if night:
        # night or "secret" look: dark ink ground with a lighter dot layer of the same family
        printed[:] = hex_to_rgb(palette["ink"])
        printed[halftone_mask(1 - darkness, step, screen_angles[0])] = hex_to_rgb(mix_colors(palette["ink"], palette["paper"], 0.4))
        printed[halftone_mask(np.clip((0.3 - darkness) * 3, 0, 1), step, screen_angles[1])] = hex_to_rgb(palette["highlight"])
        return Image.fromarray(printed.astype(np.uint8))
    printed[:] = hex_to_rgb(palette["paper"])
    ink_layers = [
        (palette["highlight"], np.clip(darkness * 1.1, 0, 0.85), screen_angles[0], (0, 0)),
        (palette["shadow"], np.clip((darkness - 0.3) * 1.6, 0, 1), screen_angles[1], (2, 3)),
        (palette["ink"], np.clip((darkness - 0.6) * 2.4, 0, 1), screen_angles[2], (-1, 1)),
    ]
    for ink_hex, coverage, angle, register_offset in ink_layers:
        dots = np.roll(halftone_mask(coverage, step, angle), register_offset, axis=(0, 1))
        # overprint: where inks land on each other they multiply, like real print
        printed[dots] = printed[dots] * hex_to_rgb(ink_hex) / 255
    return Image.fromarray(printed.astype(np.uint8))


# ---------- scissor-cut edges ----------
# resamples one outline every ~9.5 px and pushes each point along the edge's normal by slow noise,
# plus a rare 1-2 px notch where the scissors slipped
def cut_contour(points, noise_maker, amplitude):
    closed_points = np.vstack([points, points[:1]])
    segment_lengths = np.hypot(*np.diff(closed_points, axis=0).T)
    distance_along = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = distance_along[-1]
    sample_count = max(6, int(total_length / 9.5))
    sample_positions = np.linspace(0, total_length, sample_count, endpoint=False)
    x = np.interp(sample_positions, distance_along, closed_points[:, 0])
    y = np.interp(sample_positions, distance_along, closed_points[:, 1])
    # direction along the edge, turned 90 degrees, is the normal
    tangent_x = np.roll(x, -1) - np.roll(x, 1)
    tangent_y = np.roll(y, -1) - np.roll(y, 1)
    tangent_length = np.hypot(tangent_x, tangent_y) + 1e-6
    normal_x, normal_y = tangent_y / tangent_length, -tangent_x / tangent_length
    # low-frequency noise: three slow waves along the edge with random lengths and phases
    wobble = np.zeros(sample_count)
    for _ in range(3):
        wavelength = noise_maker.uniform(40, 160)
        wobble += np.sin(2 * np.pi * sample_positions / wavelength + noise_maker.uniform(0, 2 * np.pi))
    wobble = wobble / 3 * amplitude * 1.6
    notches = noise_maker.random(sample_count) < 0.025
    wobble[notches] -= noise_maker.uniform(1, 2, notches.sum())
    cut_points = np.stack([x + normal_x * wobble, y + normal_y * wobble], axis=1)
    return np.round(cut_points).astype(np.int32)


# cuts every outline (holes too) of a 0/255 mask. Bigger shapes get a bigger wobble (1.2 to 2.2 px)
def cut_mask_edges(mask, seed):
    noise_maker = np.random.default_rng(seed)
    amplitude = 1.2 + 1.0 * min(1.0, max(mask.shape) / 900)
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cut_outlines = [cut_contour(contour[:, 0, :].astype(np.float32), noise_maker, amplitude) for contour in contours if len(contour) >= 6]
    cut_mask = np.zeros_like(mask)
    cv2.fillPoly(cut_mask, cut_outlines, 255)
    return cut_mask


# builds one paper piece as an RGBA sprite. The edge is cut ONCE here in the piece's own space and kept,
# so a moving piece keeps the same edge, exactly like real cut-out animation.
#   shape_mask: 0/255 array of the shape      fill_hex: flat ink of the piece
#   content: optional picture (array) printed on the piece instead of the flat fill
#   dot_hex / dot_coverage: a darker halftone overprinted on the piece, dots growing toward the bottom
#   outline_width: heavy ink outline (0 for none)   gap: cut gap in the ground colour   shadow: small paper shadow
def make_cut_piece(shape_mask, fill_hex, palette, seed, outline_width=0, content=None, dot_hex=None, dot_coverage=0.0, dot_step=22, dot_angle=45, gap=3, gap_hex=None, shadow=True):
    mask = np.pad(shape_mask.astype(np.uint8), CUT_PADDING)
    cut_mask = cut_mask_edges(mask, seed)
    height, width = cut_mask.shape
    piece = np.empty((height, width, 3), dtype=np.float32)
    piece[:] = hex_to_rgb(fill_hex)
    if content is not None:
        content_height, content_width = content.shape[:2]
        piece[CUT_PADDING:CUT_PADDING + content_height, CUT_PADDING:CUT_PADDING + content_width] = content
    if dot_hex and dot_coverage > 0:
        coverage = np.repeat(np.linspace(dot_coverage * 0.25, dot_coverage, height, dtype=np.float32)[:, None], width, axis=1)
        dots = halftone_mask(coverage, dot_step, dot_angle)
        piece[dots] = piece[dots] * hex_to_rgb(dot_hex) / 255
    if outline_width:
        outlines, _ = cv2.findContours(cut_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        # drawn on the edge, so half of it falls outside the piece and is cut away
        cv2.drawContours(piece, outlines, -1, tuple(float(value) for value in hex_to_rgb(palette["ink"])), outline_width * 2, cv2.LINE_AA)
    gap_mask = cv2.dilate(cut_mask, np.ones((gap * 2 + 1, gap * 2 + 1), np.uint8)) if gap else cut_mask
    sprite = np.zeros((height, width, 4), dtype=np.uint8)
    if shadow:
        # small paper shadow: offset (5, 7) px, blurred, 32%, in the dark ink
        shadow_alpha = cv2.GaussianBlur(np.roll(gap_mask, (7, 5), axis=(0, 1)).astype(np.float32), (0, 0), 4) * 0.32
        sprite[:, :, :3] = hex_to_rgb(palette["ink"]).astype(np.uint8)
        sprite[:, :, 3] = shadow_alpha.astype(np.uint8)
    sprite[gap_mask > 0, :3] = hex_to_rgb(gap_hex or palette["paper"]).astype(np.uint8)
    sprite[gap_mask > 0, 3] = 255
    sprite[cut_mask > 0, :3] = np.clip(piece[cut_mask > 0], 0, 255).astype(np.uint8)
    sprite[cut_mask > 0, 3] = 255
    return Image.fromarray(sprite, "RGBA")


# a 0/255 mask of some text, cropped tight
def make_text_mask(text, font):
    left, top, right, bottom = font.getbbox(text)
    mask_image = Image.new("L", (right - left + 4, bottom - top + 4), 0)
    ImageDraw.Draw(mask_image).text((2 - left, 2 - top), text, font=font, fill=255)
    return np.array(mask_image)


# a cut paper label with text printed on it
def make_label(text, font, fill_hex, text_hex, palette, seed):
    left, top, right, bottom = font.getbbox(text)
    padding = int(font.size * 0.45)
    label_mask = np.full((bottom - top + padding * 2, right - left + padding * 2), 255, np.uint8)
    sprite = make_cut_piece(label_mask, fill_hex, palette, seed)
    ImageDraw.Draw(sprite).text((CUT_PADDING + padding - left, CUT_PADDING + padding - top), text, font=font, fill=text_hex)
    return sprite


# ---------- stamps ----------
# a double rounded rectangle and bold text, pushed through a displacement filter (rough edges)
# and a coarse ink grain (mottled ink, blotches where the stamp did not take)
def make_stamp(text, font, ink_hex, seed):
    noise_maker = np.random.default_rng(seed)
    text_left, text_top, text_right, text_bottom = font.getbbox(text)
    text_width, text_height = text_right - text_left, text_bottom - text_top
    border = max(5, text_height // 6)
    padding_x, padding_y = int(text_height * 0.7), int(text_height * 0.55)
    box_width, box_height = text_width + padding_x * 2, text_height + padding_y * 2
    margin = 20
    mask_image = Image.new("L", (box_width + margin * 2, box_height + margin * 2), 0)
    mask_draw = ImageDraw.Draw(mask_image)
    mask_draw.rounded_rectangle([margin, margin, margin + box_width, margin + box_height], radius=box_height * 0.22, outline=255, width=border)
    inset = border * 2
    mask_draw.rounded_rectangle([margin + inset, margin + inset, margin + box_width - inset, margin + box_height - inset], radius=box_height * 0.15, outline=255, width=max(2, border // 2))
    mask_draw.text((margin + padding_x - text_left, margin + padding_y - text_top), text, font=font, fill=255)
    ink_amount = np.array(mask_image, dtype=np.float32)
    height, width = ink_amount.shape
    rows, columns = np.mgrid[0:height, 0:width].astype(np.float32)
    # displacement: slow random offsets push the ink around
    shift_x = cv2.resize(noise_maker.uniform(-1, 1, (height // 8 + 2, width // 8 + 2)).astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC) * 2.5
    shift_y = cv2.resize(noise_maker.uniform(-1, 1, (height // 8 + 2, width // 8 + 2)).astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC) * 2.5
    ink_amount = cv2.remap(ink_amount, columns + shift_x, rows + shift_y, cv2.INTER_LINEAR)
    # coarse ink grain: blotches with no ink, plus fine pin holes
    blotches = cv2.resize(noise_maker.uniform(0, 1, (height // 4 + 2, width // 4 + 2)).astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    inked = (ink_amount > 110) & (blotches > 0.2) & (noise_maker.random((height, width)) > 0.05)
    sprite = np.zeros((height, width, 4), dtype=np.uint8)
    sprite[inked, :3] = hex_to_rgb(ink_hex).astype(np.uint8)
    sprite[inked, 3] = 255
    return Image.fromarray(sprite, "RGBA")


# ---------- props: flat fill + ink outline, all hand cut ----------
# names: starburst, speech_bubble, sticky_note, folder, clock. label_text is printed inside a speech bubble
def make_prop(prop_name, size, palette, seed, label_text="", label_font=None):
    mask_image = Image.new("L", (size, size), 0)
    mask_draw = ImageDraw.Draw(mask_image)
    outline_width = max(3, size // 45)
    if prop_name == "starburst":
        spike_points = []
        for point_index in range(36):
            radius = size * (0.48 if point_index % 2 == 0 else 0.33)
            angle = point_index * math.pi / 18
            spike_points.append((size / 2 + math.cos(angle) * radius, size / 2 + math.sin(angle) * radius))
        mask_draw.polygon(spike_points, fill=255)
        fill_hex = palette["highlight"]
    elif prop_name == "speech_bubble":
        mask_draw.ellipse([size * 0.05, size * 0.08, size * 0.95, size * 0.7], fill=255)
        mask_draw.polygon([(size * 0.3, size * 0.6), (size * 0.48, size * 0.64), (size * 0.16, size * 0.94)], fill=255)
        fill_hex = palette["paper"]
    elif prop_name == "sticky_note":
        mask_draw.polygon([(size * 0.08, size * 0.08), (size * 0.92, size * 0.08), (size * 0.92, size * 0.74), (size * 0.76, size * 0.92), (size * 0.08, size * 0.92)], fill=255)
        fill_hex = palette["highlight"]
    elif prop_name == "folder":
        mask_draw.polygon([(size * 0.05, size * 0.2), (size * 0.38, size * 0.2), (size * 0.45, size * 0.3), (size * 0.95, size * 0.3), (size * 0.95, size * 0.85), (size * 0.05, size * 0.85)], fill=255)
        fill_hex = palette["spot"]
    else:
        mask_draw.ellipse([size * 0.06, size * 0.06, size * 0.94, size * 0.94], fill=255)
        fill_hex = palette["paper"]
    sprite = make_cut_piece(np.array(mask_image), fill_hex, palette, seed, outline_width=outline_width)
    # ink details on top. Everything is offset by the empty space around the piece
    detail_draw = ImageDraw.Draw(sprite)
    offset = CUT_PADDING
    ink_hex = palette["ink"]
    if prop_name == "speech_bubble" and label_text:
        left, top, right, bottom = label_font.getbbox(label_text)
        detail_draw.text((offset + size * 0.5 - (right + left) / 2, offset + size * 0.39 - (bottom + top) / 2), label_text, font=label_font, fill=ink_hex)
    elif prop_name == "sticky_note":
        for line_index in range(3):
            line_y = offset + size * (0.3 + line_index * 0.16)
            detail_draw.line([(offset + size * 0.2, line_y), (offset + size * (0.78 - line_index * 0.12), line_y + size * 0.02)], fill=ink_hex, width=outline_width)
        # folded corner
        detail_draw.polygon([(offset + size * 0.76, offset + size * 0.92), (offset + size * 0.76, offset + size * 0.74), (offset + size * 0.92, offset + size * 0.74)], fill=palette["shadow"], outline=ink_hex)
    elif prop_name == "folder":
        detail_draw.line([(offset + size * 0.05, offset + size * 0.42), (offset + size * 0.95, offset + size * 0.42)], fill=ink_hex, width=outline_width)
        detail_draw.rectangle([offset + size * 0.38, offset + size * 0.55, offset + size * 0.62, offset + size * 0.65], fill=palette["paper"], outline=ink_hex, width=max(2, outline_width // 2))
    elif prop_name == "clock":
        for tick_index in range(12):
            angle = tick_index * math.pi / 6
            inner, outer = size * 0.32, size * 0.38
            detail_draw.line([(offset + size / 2 + math.cos(angle) * inner, offset + size / 2 + math.sin(angle) * inner), (offset + size / 2 + math.cos(angle) * outer, offset + size / 2 + math.sin(angle) * outer)], fill=ink_hex, width=outline_width)
    return sprite


# a little paper explorer: wide brim hat, one big round eye, a flared coat, a scarf that flaps and thin legs with round boots.
# pose 0 and 1 swap the legs and flap the scarf for a walk cycle
def make_character(size, fill_hex, palette, seed, pose=0):
    mask_image = Image.new("L", (size, size), 0)
    mask_draw = ImageDraw.Draw(mask_image)
    # round head
    mask_draw.ellipse([size * 0.35, size * 0.12, size * 0.65, size * 0.4], fill=255)
    # hat: wide brim plus a tall crown
    mask_draw.ellipse([size * 0.2, size * 0.13, size * 0.8, size * 0.23], fill=255)
    mask_draw.rounded_rectangle([size * 0.36, size * 0.01, size * 0.64, size * 0.18], radius=size * 0.03, fill=255)
    # coat that flares out like a bell
    mask_draw.polygon([(size * 0.38, size * 0.38), (size * 0.62, size * 0.38), (size * 0.8, size * 0.76), (size * 0.2, size * 0.76)], fill=255)
    # scarf tail flapping behind, it lifts higher on pose 1
    scarf_tip_y = size * (0.5 if pose == 0 else 0.4)
    mask_draw.polygon([(size * 0.4, size * 0.38), (size * 0.08, scarf_tip_y), (size * 0.12, scarf_tip_y + size * 0.08), (size * 0.42, size * 0.46)], fill=255)
    # thin legs, apart on pose 0 and together on pose 1, each with a round boot
    if pose == 0:
        leg_x_positions = [0.36, 0.6]
    else:
        leg_x_positions = [0.44, 0.52]
    for leg_x in leg_x_positions:
        mask_draw.rectangle([size * leg_x, size * 0.74, size * (leg_x + 0.05), size * 0.92], fill=255)
        mask_draw.ellipse([size * (leg_x - 0.03), size * 0.89, size * (leg_x + 0.12), size * 0.99], fill=255)
    sprite = make_cut_piece(np.array(mask_image), fill_hex, palette, seed, outline_width=max(5, size // 26), dot_hex=palette["ink"], dot_coverage=0.25, dot_step=14)
    # ink details on top. Everything is offset by the empty space around the piece
    detail_draw = ImageDraw.Draw(sprite)
    offset = CUT_PADDING
    ink_hex = palette["ink"]
    # the hat is solid ink with a band in the highlight colour
    detail_draw.ellipse([offset + size * 0.2, offset + size * 0.13, offset + size * 0.8, offset + size * 0.23], fill=ink_hex)
    detail_draw.rounded_rectangle([offset + size * 0.36, offset + size * 0.01, offset + size * 0.64, offset + size * 0.18], radius=size * 0.03, fill=ink_hex)
    detail_draw.rectangle([offset + size * 0.36, offset + size * 0.13, offset + size * 0.64, offset + size * 0.16], fill=palette["highlight"])
    # one big round eye looking ahead
    detail_draw.ellipse([offset + size * 0.45, offset + size * 0.24, offset + size * 0.6, offset + size * 0.37], fill=palette["paper"], outline=ink_hex, width=max(2, size // 60))
    detail_draw.ellipse([offset + size * 0.52, offset + size * 0.28, offset + size * 0.58, offset + size * 0.34], fill=ink_hex)
    # scarf in the alarm red and three coat buttons
    detail_draw.polygon([(offset + size * 0.4, offset + size * 0.38), (offset + size * 0.08, offset + scarf_tip_y), (offset + size * 0.12, offset + scarf_tip_y + size * 0.08), (offset + size * 0.42, offset + size * 0.46)], fill=palette["alarm"], outline=ink_hex)
    for button_index in range(3):
        button_y = size * (0.5 + button_index * 0.08)
        detail_draw.ellipse([offset + size * 0.48, offset + button_y, offset + size * 0.52, offset + button_y + size * 0.04], fill=ink_hex)
    return sprite


# sun rays: wedges of one flat ink, made at half size (they are blown up after turning)
def make_rays(size, ray_hex, palette, seed, ray_count=16):
    mask_image = Image.new("L", (size, size), 0)
    mask_draw = ImageDraw.Draw(mask_image)
    center = size / 2
    for ray_index in range(ray_count):
        first_angle = 2 * math.pi * ray_index / ray_count
        second_angle = first_angle + math.pi / ray_count
        mask_draw.polygon([(center, center), (center + math.cos(first_angle) * size, center + math.sin(first_angle) * size), (center + math.cos(second_angle) * size, center + math.sin(second_angle) * size)], fill=255)
    return make_cut_piece(np.array(mask_image), ray_hex, palette, seed, gap=0, shadow=False)


# small cut paper confetti in the spot inks
def make_confetti(palette, size, piece_count, seed):
    confetti_colors = [palette["highlight"], palette["shadow"], palette["spot"], palette["alarm"]]
    pieces = []
    for piece_index in range(piece_count):
        mask_image = Image.new("L", (size, size), 0)
        mask_draw = ImageDraw.Draw(mask_image)
        shape_kind = piece_index % 3
        if shape_kind == 0:
            mask_draw.rectangle([0, size * 0.25, size, size * 0.75], fill=255)
        elif shape_kind == 1:
            mask_draw.polygon([(size * 0.5, 0), (size, size), (0, size)], fill=255)
        else:
            mask_draw.ellipse([size * 0.1, size * 0.1, size * 0.9, size * 0.9], fill=255)
        pieces.append(make_cut_piece(np.array(mask_image), confetti_colors[piece_index % len(confetti_colors)], palette, seed + piece_index, gap=1, shadow=False))
    return pieces
