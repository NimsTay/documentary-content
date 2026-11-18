# TO RUN: python src/make_voice.py
# Reads inputs/narrative.txt and makes one Kokoro narration clip per scene, saved as audio/audio_1.wav, audio/audio_2.wav, etc.
import io
import os
import re
import sys
from contextlib import redirect_stderr, redirect_stdout
from mlx_audio.tts.generate import generate_audio

# Folders are found relative to this file so the script works from any directory
project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The doc name is passed in as the first argument
doc_name = sys.argv[1]
audio_folder = os.path.join(project_folder, "audio", doc_name)
os.makedirs(audio_folder, exist_ok=True)

# Kokoro settings - tweak these
# voice_name = "af_nicole"  very realistic female but whispering
# voice_name = "am_echo"  best sounding man with speed 1
voice_name = "am_santa" #best sounding but a lil fruity 
language_code = "a"  
speaking_speed = 0.8  #do .8 for santa most realistic

storyboard_text = open(os.path.join(project_folder, "inputs", "narrative.txt")).read()

# Each scene starts with "scene #<number>:" and has a "* Script: <narration>" line (capital S or not)
scene_numbers = re.findall(r"scene #(\d+):", storyboard_text)
scene_scripts = re.findall(r"\* script: (.+)", storyboard_text, re.IGNORECASE)



for scene_number, scene_script in zip(scene_numbers, scene_scripts):
    clip_name = "scene_" + scene_number
    # Hide all the library's own output (pipeline, text, progress bars) while the clip is made
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        generate_audio(text=scene_script, model="prince-canuma/Kokoro-82M", voice=voice_name, lang_code=language_code, speed=speaking_speed, output_path=audio_folder, file_prefix=clip_name, audio_format="wav", join_audio=True, verbose=False)
    print("Completed scene", scene_number)

print("Done! Clips saved in", audio_folder)


# american_voices = ['am_puck', 'am_onyx', 'am_santa', ]
# british_voices = ['bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis']
# american_female_voices = ['af_heart', 'af_bella', 'af_nicole', 'af_sarah', 'af_sky', 'af_aoede', 'af_jessica', 'af_kore', 'af_nova', 'af_river', 'af_alloy']
# british_female_voices = ['bf_emma', 'bf_alice', 'bf_isabella', 'bf_lily']
# for voice in american_voices:
#     clip_name = "audio_" + voice
#     with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
#         generate_audio(text=scene_scripts[0], model="prince-canuma/Kokoro-82M", voice=voice, lang_code='a', speed=0.95, output_path=audio_folder, file_prefix=clip_name, audio_format="wav", join_audio=True, verbose=False)

# for voice in american_female_voices:
#     clip_name = "audio_" + voice
#     with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
#         generate_audio(text=scene_scripts[0], model="prince-canuma/Kokoro-82M", voice=voice, lang_code='a', speed=0.95, output_path=audio_folder, file_prefix=clip_name, audio_format="wav", join_audio=True, verbose=False)

# for voice in british_voices:
#     clip_name = "audio_" + voice
#     with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
#         generate_audio(text=scene_scripts[0], model="prince-canuma/Kokoro-82M", voice=voice, lang_code='b', speed=0.95, output_path=audio_folder, file_prefix=clip_name, audio_format="wav", join_audio=True, verbose=False)

# for voice in british_female_voices:
#     clip_name = "audio_" + voice
#     with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
#         generate_audio(text=scene_scripts[0], model="prince-canuma/Kokoro-82M", voice=voice, lang_code='b', speed=0.95, output_path=audio_folder, file_prefix=clip_name, audio_format="wav", join_audio=True, verbose=False)

