import os
import csv
from PIL import Image

def get_box_path(wid, dl_dir="/app/data/images", has_header=True):
    box_dir = os.path.join(dl_dir, wid[-2:] if has_header else "")
    path = os.path.join(box_dir, wid + ".png")
    return box_dir, path

def load_worlds(path="/app/data/best_worlds.csv", has_description=False):
    worlds = []
    with open(path, "r") as f:
        for items in csv.reader(f):
            try:
                if len(items) < 4 or items[0].startswith("#"):
                    continue
                box = {
                    "id": items[0],
                    "author": items[1],
                    "title": items[2],
                    "favorites": int(items[3]),
                    "image_url": items[4],
                }
                if has_description:
                    box["description"] = items[5]
                worlds.append(box)
            except:
                print("IGNORED LINE: " + str(items))
    return worlds

def load_good_image(path, resize=256):
    img = load_resized_image(path, resize=resize)
    left = (img.width - resize) // 2
    upper = (img.height - resize) // 2
    right = left + resize
    lower = upper + resize
    return img.crop((left, upper, right, lower))

def load_resized_image(path, resize=256):
    img = Image.open(path)
    img = img.convert("RGB")
    w, h = img.size
    r = resize / min(w, h)
    return img.resize((int(w*r),int(h*r)), Image.Resampling.LANCZOS)
