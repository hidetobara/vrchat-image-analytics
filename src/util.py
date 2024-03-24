import os
import csv
from PIL import Image


def load_worlds(path="/app/data/best_worlds.csv", ignore_no_image=False):
    worlds = []
    with open(path, "r") as f:
        for items in csv.reader(f):
            if len(items) < 5 or items[0].startswith("#"):
                continue
            wid = items[0]
            img_path = os.path.join("/app/data/images", wid + ".png")
            if ignore_no_image and not os.path.exists(img_path):
                continue
            author = items[1]
            title = items[2]
            image_url = items[4]
            worlds.append({"id": wid, "author": author, "title": title, "image_url": image_url})
    return worlds

def load_good_image(path):
    img = Image.open(path)
    img = img.convert("RGB")
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    return img
