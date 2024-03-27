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

def load_good_image(path, resize=224):
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
