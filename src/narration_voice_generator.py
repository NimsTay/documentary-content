# TO RUN: python src/make_voice.py
# Reads the narrative file (second argument, e.g. inputs/<doc_name>/narrative.txt) and makes one ElevenLabs narration clip per scene, saved as audio/<doc_name>/scene_1.mp3, scene_2.mp3, etc.
import os
import re
import sys
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

# The ElevenLabs key is read from the .env file, like the Krea key in image_generator.py
load_dotenv()
elevenlabs_client = ElevenLabs(api_key=os.environ["ELEVEN_LABS_KEY"])

# Folders are found relative to this file so the script works from any directory
project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The doc name is passed in as the first argument
doc_name = sys.argv[1]
narrative_path = sys.argv[2]
audio_folder = os.path.join(project_folder, "audio", doc_name)
os.makedirs(audio_folder, exist_ok=True)

# ElevenLabs settings - tweak these
model_name = "eleven_v4_turbo"
# Jon - Warm & Grounded Storyteller (other Jon voices have different ids, copy the id from the ElevenLabs Voice Library)
jon_voice_id = "JYUzXNeyCPmymMkfcicu"

storyboard_text = open(narrative_path).read()

# Each scene starts with "scene #<number>:" and has a "* Script: <narration>" line (capital S or not)
scene_numbers = re.findall(r"scene #(\d+)", storyboard_text, re.IGNORECASE)
scene_scripts = re.findall(r"\* script: (.+)", storyboard_text, re.IGNORECASE)



for scene_number, scene_script in zip(scene_numbers, scene_scripts):
    clip_name = "scene_" + scene_number
    # ElevenLabs sends the audio back in chunks, so join them and save as an mp3 file
    audio_chunks = elevenlabs_client.text_to_speech.convert(text=scene_script, voice_id=jon_voice_id, model_id=model_name)
    open(os.path.join(audio_folder, clip_name + ".mp3"), "wb").write(b"".join(audio_chunks))
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

