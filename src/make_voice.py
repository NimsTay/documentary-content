# TO RUN: python src/make_voice.py
# Reads inputs/storyboard.txt and makes one Kokoro narration clip per scene, saved as audio/audio_1.wav, audio/audio_2.wav, etc.
import io
import os
import re
from contextlib import redirect_stderr, redirect_stdout
from mlx_audio.tts.generate import generate_audio

# Folders are found relative to this file so the script works from any directory
project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
audio_folder = os.path.join(project_folder, "audio")
os.makedirs(audio_folder, exist_ok=True)

# Kokoro settings - tweak these
# voice_name = "af_nicole"  very realistic female but whispering
voice_name = "am_michael"  # #best sounding man good at speed 1.1 or 1
language_code = "a"  
speaking_speed = 1.1  # below 1.0 is slower and more dramatic, above 1.0 is faster

storyboard_text = open(os.path.join(project_folder, "inputs", "storyboard.txt")).read()

# Each scene starts with "scene #<number>:" and has a "* Script: <narration>" line (capital S or not)
scene_numbers = re.findall(r"scene #(\d+):", storyboard_text)
scene_scripts = re.findall(r"\* script: (.+)", storyboard_text, re.IGNORECASE)

for scene_number, scene_script in zip(scene_numbers, scene_scripts):
    clip_name = "audio_" + scene_number

    # Hide all the library's own output (pipeline, text, progress bars) while the clip is made
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        generate_audio(text=scene_script, model="prince-canuma/Kokoro-82M", voice=voice_name, lang_code=language_code, speed=speaking_speed, output_path=audio_folder, file_prefix=clip_name, audio_format="wav", join_audio=True, verbose=False)

    print("Completed scene", scene_number)

print("Done! Clips saved in", audio_folder)
