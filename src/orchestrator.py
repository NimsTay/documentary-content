# Runs the whole documentary pipeline for one doc name.
# TO RUN (from the documentary-content folder): python src/orchestrator.py <doc_name>
import glob
import subprocess
import sys

doc_name = sys.argv[1]

# Each project has its own folder in inputs/ with its own narrative.txt
narrative_path = "inputs/" + doc_name + "/narrative.txt"

# Each step is a script in src/ that takes the doc name and the narrative path as arguments, run in order
steps = [
    # "image_generator.py",
    # "image_coords_extractor.py",  # needed by the parallax animator
    # "depth_map_generator.py",  # needed by the parallax animator
    "narration_voice_generator.py",
    "animation_parallax_generator.py",  # only makes the scenes in parallax_scenes
    "animate_image_with_krea.py",  # makes every scene EXCEPT the ones in parallax_scenes
    "final_video_generator.py",
]

# parallax_scenes = ["scene_2", "scene_5", "scene_8", "scene_13"] #yuri
parallax_scenes = ["scene_2", "scene_3", "scene_6", "scene_9", "scene_10", "scene_11","scene_13","scene_16","scene_17","scene_19","scene_20","scene_21","scene_22","scene_25","scene_27" ] #gta

# The two animators get the parallax scene list as extra arguments (the parallax one makes only those, the Krea one skips them)
steps_that_need_parallax_scenes = ["animation_parallax_generator.py", "animate_image_with_krea.py"]

try:
    for step in steps:
        print('==========================================')
        print("Running", step)
        print('==========================================\n\n\n')

        extra_arguments = parallax_scenes if step in steps_that_need_parallax_scenes else []
        subprocess.run([sys.executable, "src/" + step, doc_name, narrative_path] + extra_arguments, check=True)
finally:
    # Runs even if a step crashes: kill leftover ffmpeg rawvideo processes so they don't pile up in memory
    subprocess.run(["pkill", "-9", "-f", "ffmpeg -y -loglevel error -f rawvideo"])

    # Also kill any leftover python process running a script in src/, except this orchestrator
    for script_path in glob.glob("src/*.py"):
        if "orchestrator.py" not in script_path:
            subprocess.run(["pkill", "-9", "-f", script_path])

print("Done! Final video is in videos/" + doc_name + "/final_video.mp4")



#the tidbit archive
