# Runs the whole documentary pipeline for one doc name.
# TO RUN (from the documentary-content folder): python src/orchestrator.py <doc_name>
import subprocess
import sys

doc_name = sys.argv[1]

# Each step is a script in src/ that takes the doc name as its argument, run in order
steps = [
    # "image_generator.py",
    # "image_coords_extractor.py",
    # "depth_map_generator.py",
    # "narration_voice_generator.py",
    "animation_parallax_generator.py",
    "final_video_generator.py",
]

for step in steps:
    print('==========================================')
    print("Running", step)
    print('==========================================\n\n\n')
    
    subprocess.run([sys.executable, "src/" + step, doc_name], check=True)

print("Done! Final video is in videos/" + doc_name + "/final_video.mp4")
