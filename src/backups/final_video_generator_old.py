# TO RUN: python src/final_video_generator.py <doc_name>
# Joins videos/<doc_name>/scene_1.mp4, scene_2.mp4, ... in order into one big video: videos/<doc_name>/final_video.mp4
# The clips are just copied back to back (no re-encoding), so every clip keeps its own timing. All clips must have the same frame rate (24).
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

# Find every scene_<number>.mp4 and sort by number so scene_2 comes before scene_10
clip_names = [name for name in os.listdir(videos_folder) if re.fullmatch(r"scene_\d+\.mp4", name)]
clip_names.sort(key=lambda name: int(re.findall(r"\d+", name)[0]))

# ffmpeg's concat needs a text file listing the clips, one "file '<path>'" line per clip
with open(clip_list_path, "w") as clip_list_file:
    for clip_name in clip_names:
        clip_list_file.write("file '" + os.path.join(videos_folder, clip_name) + "'\n")

# Append the clips back to back, copying them as they are
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", clip_list_path, "-c", "copy", final_video_path], check=True)

# The list file was only needed for ffmpeg
os.remove(clip_list_path)

print("Done! Joined", len(clip_names), "clips into", final_video_path)
