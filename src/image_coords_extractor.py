
import tkinter as tk
from PIL import Image, ImageTk
import os
import json
import re
import sys

# The doc name is passed in as the first argument; images are read from images/<doc>
project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDER = os.path.join(project_folder, "images", sys.argv[1])

root = tk.Tk()
root.title("Image Annotator")

files = sorted(
    [f for f in os.listdir(FOLDER) if re.match(r"scene_\d+\.(png|jpg|jpeg)$", f)],
    key=lambda f: int(re.search(r"scene_(\d+)", f).group(1))
)

files = [os.path.join(FOLDER, f) for f in files]

index = 0
data = {}
selected = None

canvas = tk.Canvas(root, bg="black")
canvas.pack(fill="both", expand=True)

label = tk.Label(root, text="")
label.pack()

def show_image():
    global img, photo, scale, selected

    selected = None
    img = Image.open(files[index])

    scale = min(1000 / img.width, 650 / img.height, 1)
    resized = img.resize((int(img.width * scale), int(img.height * scale)))
    photo = ImageTk.PhotoImage(resized)

    canvas.config(width=resized.width, height=resized.height)
    canvas.delete("all")
    canvas.create_image(0, 0, anchor="nw", image=photo)

    label.config(text=f"{index + 1}/{len(files)} - {os.path.basename(files[index])}")

def click(event):
    global selected

    x = int(event.x / scale)
    y = int(event.y / scale)
    selected = (x, y)

    canvas.delete("marker")
    canvas.create_oval(
        event.x-5, event.y-5, event.x+5, event.y+5,
        fill="red", tags="marker"
    )

    label.config(text=f"Selected: x={x}, y={y}")

def next_image():
    global index

    name = os.path.basename(files[index].replace('.png', '').replace('.jpg', '').replace('.jpeg', ''))
    data[name] = {"x": selected[0], "y": selected[1]}

    with open(os.path.join(FOLDER, "focal_coords.json"), "w") as f:
        json.dump(data, f, indent=4)

    index += 1

    if index < len(files):
        show_image()
    else:
        # Last image saved, so close the window which ends the script
        root.destroy()

canvas.bind("<Button-1>", click)

tk.Button(root, text="Save & Next", command=next_image).pack()

show_image()
root.mainloop()
