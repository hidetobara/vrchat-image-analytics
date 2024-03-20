import argparse
import csv
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AutoProcessor, CLIPModel, AutoTokenizer
import numpy


MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_TRAINED = "/app/models/vrchat"

model = CLIPModel.from_pretrained(MODEL_TRAINED).to(DEVICE)
processor = AutoProcessor.from_pretrained(MODEL_NAME)


def load_worlds(path="/app/data/best_worlds.csv"):
    worlds = []
    with open(path, "r") as f:
        for items in csv.reader(f):
            if len(items) < 4 or items[0].startswith("#"):
                continue
            wid = items[0]
            title = items[1]
            image_url = items[2]
            worlds.append({"id": wid,  "title": title, "image_url": image_url})
    return worlds

def search_title(image_path, worlds):
    img = Image.open(image_path)
    img = img.convert("RGB")
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    titles = list(map(lambda x: x["title"], worlds))

    inputs = processor(text=titles, images=img, padding=True, truncation=True, max_length=32, return_tensors="pt")

    inputs = inputs.to(DEVICE)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).cpu()
    probs = torch.reshape(probs, [-1])
    idx = torch.argmax(probs)
    print("P=", probs[idx], "Title=", titles[idx])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Let's evaluate !")
    parser.add_argument('--image', help="an image path for evaluate")
    args = parser.parse_args()

    worlds = load_worlds()

    if args.image:
        search_title(args.image, worlds)