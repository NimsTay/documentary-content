# Reads storyboard.txt and makes two narration clips per scene (Kokoro and Chatterbox) in the audio/ folder
import re
from mlx_audio.tts.generate import generate_audio

# Where the storyboard is read from and where the clips are saved
storyboard_file = "storyboard.txt"
audio_folder = "audio"


def make_kokoro_clip(scene_script, clip_name):
    # Kokoro settings - tweak these
    voice_name = "bm_lewis"  # British male voices: bm_george, bm_lewis, bm_fable, bm_daniel
    language_code = "b"  # "b" is British English, "a" is American English
    speaking_speed = 0.85  # below 1.0 is slower and more dramatic, above 1.0 is faster

    generate_audio(text=scene_script, model="prince-canuma/Kokoro-82M", voice=voice_name, lang_code=language_code, speed=speaking_speed, output_path=audio_folder, file_prefix=clip_name + "_kokoro", audio_format="wav", join_audio=True, verbose=False)


def make_chatterbox_clip(scene_script, clip_name):
    # Chatterbox settings - tweak these
    emotion_exaggeration = 0.7  # 0.1 is flat, 0.5 is normal, 1.0 and up is very dramatic
    pacing_strictness = 0.1  # lower (0.1-0.3) is slower and more deliberate, higher (0.5+) is faster
    randomness = 0.6  # lower is more consistent, higher is more varied and expressive
    reference_voice_file = "my_voice.wav"  # 10-20 second wav clip to copy a voice from, or None for the default voice

    generate_audio(text=scene_script, model="mlx-community/chatterbox-fp16", ref_audio=reference_voice_file, exaggeration=emotion_exaggeration, cfg_weight=pacing_strictness, temperature=randomness, output_path=audio_folder, file_prefix=clip_name + "_chatterbox", audio_format="wav", join_audio=True, verbose=False)


storyboard_text = open(storyboard_file).read()

# Each scene starts with "scene #<number>:" and has a "* script: <narration>" line
scene_numbers = re.findall(r"scene #(\d+):", storyboard_text)
scene_scripts = re.findall(r"\* script: (.+)", storyboard_text)

for scene_number, scene_script in zip(scene_numbers, scene_scripts):
    clip_name = "scene_" + scene_number.zfill(2)
    print("Making", clip_name)
    make_kokoro_clip(scene_script, clip_name)
    # make_chatterbox_clip(scene_script, clip_name)
    exit()

print("Done! Clips saved in", audio_folder + "/")
