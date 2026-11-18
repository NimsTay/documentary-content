# Makes one Krea image per scene in inputs/narrative.txt and saves them in images/
import os
import re
import sys
import time
import requests
from dotenv import load_dotenv

load_dotenv()
krea_api_key = os.environ["KREA_KEY"]

generate_url = "https://api.krea.ai/generate/image/z-image/z-image"
jobs_url = "https://api.krea.ai/jobs/"
auth_headers = {"Authorization": f"Bearer {krea_api_key}"}

# Folders are found relative to this file so the script works from any directory
project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The doc name is passed in as the first argument
doc_name = sys.argv[1]
images_folder = os.path.join(project_folder, "images", doc_name)
os.makedirs(images_folder, exist_ok=True)

# Split the storyboard into one block of text per scene
storyboard_text = open(os.path.join(project_folder, "inputs", "narrative.txt")).read()
scene_blocks = re.split(r"(?mi)^scene #", storyboard_text)[1:]  # case insensitive so "Scene #1" and "scene #1" both work


# Generates an image with the Krea z-image model from the scene text, returns the image bytes (or None)
def generate_krea_image(scene_block):
    # The prompt is the whole scene block exactly as written in the storyboard (title, script, image, motion)
    prompt = "scene #" + scene_block.strip() + "\n"

    request_body = {
        "prompt": prompt,
        "denoising_strength": 0.8,
        "skip_prompt_expansion": False,
        "aspect_ratio": "16:9",
        "resolution": "1K",
        # "seed": 1337,
    }

    job_id = requests.post(generate_url, headers=auth_headers, json=request_body).json()["job_id"]

    # Check the job every few seconds until it is finished
    while True:
        job = requests.get(jobs_url + job_id, headers=auth_headers).json()
        print(f"job {job_id} status: {job['status']}")
        if job["status"] in ["completed", "failed", "cancelled"]:
            break
        time.sleep(3)

    if job["status"] != "completed":
        return None

    return requests.get(job["result"]["urls"][0]).content


# Searches Wikimedia Commons (free stock images, no key needed) using the scene's script text, returns the image bytes (or None)
def find_stock_image(scene_block):
    # The search text is the story description from the "* script:" line
    script_line = [line for line in scene_block.splitlines() if line.startswith("* script:")][0]
    script_text = script_line.replace("* script:", "").strip()

    # Commons finds nothing for a whole paragraph, so search with the names in it (capitalized words like Ferdinand, Zeppelin)
    # Words that start a sentence are skipped because they are capitalized for no reason
    sentences = re.split(r"[.?!]\s+", script_text)
    names = []
    for sentence in sentences:
        for word in sentence.split()[1:]:
            word = word.strip(",.?!\"'")
            if word[:1].isupper() and word not in names:
                names.append(word)
    search_text = " ".join(names[:4])

    search_params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrnamespace": 6,  # 6 means files (images) on Commons
        "gsrsearch": search_text,
        "gsrlimit": 1,
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": 1280,  # ask for a 1280px wide thumbnail, Wikimedia blocks downloading huge originals
    }
    commons_headers = {"User-Agent": "documentary-image-script"}
    search_result = requests.get("https://commons.wikimedia.org/w/api.php", params=search_params, headers=commons_headers).json()

    if "query" not in search_result:
        return None

    first_page = list(search_result["query"]["pages"].values())[0]
    image_url = first_page["imageinfo"][0]["thumburl"]
    image_response = requests.get(image_url, headers=commons_headers)

    # Wikimedia answers 429 when we go too fast, so wait a bit and try again
    while image_response.status_code == 429:
        print("rate limited, waiting 30 seconds")
        time.sleep(30)
        image_response = requests.get(image_url, headers=commons_headers)

    image_response.raise_for_status()  # stops us from saving an error page as an image
    return image_response.content


for scene_block in scene_blocks:
    # The scene number is the first thing on the first line, e.g. "1" from "1" or "1: Title"
    scene_number = re.match(r"\d+", scene_block).group()

    print(f"starting scene {scene_number}")
    image_bytes = generate_krea_image(scene_block)

    if image_bytes is None:
        print(f"image failed for scene {scene_number}")
        continue

    open(os.path.join(images_folder, f"scene_{scene_number}.png"), "wb").write(image_bytes)
    print(f"saved {images_folder}/scene_{scene_number}.png")
