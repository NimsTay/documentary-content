# TO RUN: python src/final_video.py
# Joins videos/clip_1.mp4, clip_2.mp4, ... in order into one big video: videos/final_video.mp4
import os
import re
import subprocess
import sys

# Folders are found relative to this file so the script works from any directory
project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The doc name is passed in as the first argument
doc_name = sys.argv[1]
videos_folder = os.path.join(project_folder, "videos", doc_name)
final_video_path = os.path.join(videos_folder, "final_video.mp4")
clip_list_path = os.path.join(videos_folder, "clip_list.txt")

# Find every clip_<number>.mp4 and sort by number so clip_2 comes before clip_10
clip_names = [name for name in os.listdir(videos_folder) if re.fullmatch(r"scene_\d+\.mp4", name)]
clip_names.sort(key=lambda name: int(re.findall(r"\d+", name)[0]))

# ffmpeg's concat needs a text file listing the clips, one "file '<path>'" line per clip
with open(clip_list_path, "w") as clip_list_file:
    for clip_name in clip_names:
        clip_list_file.write("file '" + os.path.join(videos_folder, clip_name) + "'\n")

# Join the clips back to back (re-encoded so the joins are clean)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", clip_list_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", final_video_path], check=True)

# The list file was only needed for ffmpeg
os.remove(clip_list_path)

print("Done! Joined", len(clip_names), "clips into", final_video_path)
