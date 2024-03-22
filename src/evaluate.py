import argparse
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, CLIPModel, AutoTokenizer

import util


#MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_TRAINED = "/app/models/vrchat-worlds"
MODEL_PROCESSOR = "/app/models/processor"

model = CLIPModel.from_pretrained(MODEL_TRAINED).to(DEVICE)
processor = AutoProcessor.from_pretrained(MODEL_PROCESSOR)


def search_title(image_path, worlds):
    img = Image.open(image_path)
    img = img.convert("RGB")
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    titles = list(map(lambda x: x["author"] + " " + x["title"], worlds))

    with torch.no_grad():
        inputs = processor(text=titles, images=img, padding=True, truncation=True, max_length=32, return_tensors="pt")
        inputs = inputs.to(DEVICE)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu()
    probs = torch.reshape(probs, [-1])
    top_values, top_indices = torch.topk(probs, k=5)
    for r, idx in enumerate(top_indices):
        print("P=", top_values[r], "Title=", titles[idx])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Let's evaluate !")
    parser.add_argument('--image', help="an image path for evaluate")
    args = parser.parse_args()

    worlds = util.load_worlds()

    if args.image:
        search_title(args.image, worlds)