# Makes one image per storyboard scene and saves them in images/
# Krea AI generation is kept in a function, but the loop at the bottom uses stock images
# Usage: python image_generator.py
import os
import re
import time
import requests
from dotenv import load_dotenv

load_dotenv()
krea_api_key = os.environ["KREA_KEY"]

generate_url = "https://api.krea.ai/generate/image/krea/krea-2/medium-turbo"
jobs_url = "https://api.krea.ai/jobs/"
auth_headers = {"Authorization": f"Bearer {krea_api_key}"}

style_reference_url = "https://app-uploads.krea.ai/88a836ae-ae5f-4231-a07f-5d314513fad1/1794715500160-cef26941-399c-4d9f-8f56-72ec17a3f809-sample.png"

os.makedirs("images", exist_ok=True)

# Split the storyboard into one block of text per scene
storyboard_text = open("storyboard.txt").read()
scene_blocks = re.split(r"(?m)^scene #", storyboard_text)[1:]


# Generates an image with Krea from the scene's script and direction text, returns the image bytes (or None)
def generate_krea_image(scene_block):
    # The prompt is the script and direction lines, same format as the example
    prompt = "\n" + "\n".join(line for line in scene_block.splitlines() if line.startswith("* "))

    request_body = {
        "prompt": prompt,
        "aspect_ratio": "4:3",
        "resolution": "1K",
        "creativity": "medium",
        "intensity": 0,
        "complexity": 0,
        "movement": 0,
        "strength": 0.99,
        "image_style_references": [{"url": style_reference_url, "strength": 0.5}],
    }

    job_id = requests.post(generate_url, headers=auth_headers, json=request_body).json()["job_id"]

    # Check the job every few seconds until it is finished
    while True:
        job = requests.get(jobs_url + job_id, headers=auth_headers).json()
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
    scene_number = scene_block.split(":")[0]

    image_bytes = find_stock_image(scene_block)

    if image_bytes is None:
        print(f"no image found for scene {scene_number}")
        continue

    open(f"images/scene_{scene_number}.jpg", "wb").write(image_bytes)
    print(f"saved images/scene_{scene_number}.jpg")

    # Pause between scenes so Wikimedia does not block us for too many requests
    time.sleep(3)
