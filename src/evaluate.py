import argparse
from PIL import Image
import torch
from transformers import AutoProcessor, CLIPModel, CLIPVisionModelWithProjection

import util

#MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_TRAINED = "/app/models/vrchat-worlds"
MODEL_PROCESSOR = "/app/models/processor"


def search_title(image_path, worlds):
    model = CLIPModel.from_pretrained(MODEL_TRAINED).to(DEVICE)
    processor = AutoProcessor.from_pretrained(MODEL_PROCESSOR)

    img = util.load_good_image(image_path)
    print("IMAGE_PATH=", image_path)
    titles = list(map(lambda x: x["author"] + " " + x["title"], worlds))

    with torch.no_grad():
        inputs = processor(text=titles, images=img, padding=True, truncation=True, max_length=32, return_tensors="pt")
        inputs = inputs.to(DEVICE)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        text_embeds = outputs.text_embeds
        probs = logits_per_image.softmax(dim=1).cpu()
    probs = torch.reshape(probs, [-1])
    top_values, top_indices = torch.topk(probs, k=5)
    for r, idx in enumerate(top_indices):
        print("P=", top_values[r], "Title=", titles[idx])
    torch.save(text_embeds, "/app/data/worlds.text.pt")

def search_title_using_embeds(image_path, worlds, embeds_path):
    model = CLIPVisionModelWithProjection.from_pretrained(MODEL_TRAINED).to(DEVICE)
    processor = AutoProcessor.from_pretrained(MODEL_PROCESSOR)

    text_embeds = torch.load(embeds_path).cpu()
    #print("TEXT=", text_embeds.shape)
    img = util.load_good_image(image_path)
    print("IMAGE_PATH=", image_path)

    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt")
        inputs = inputs.to(DEVICE)
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds.cpu()
    #print("IMAGE=", image_embeds.shape)
    dist = torch.cdist(text_embeds, image_embeds)
    dist = torch.reshape(dist, [-1])
    indices = torch.argsort(dist)
    for r in range(5):
        idx = indices[r]
        print("DIST=", dist[idx], "IDX=", idx, "TITLE=", worlds[idx])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Let's evaluate !")
    parser.add_argument('--image', help="an image path for evaluate")
    parser.add_argument('--text_embeds', help="using text embeds")
    args = parser.parse_args()

    if args.image:
        worlds = util.load_worlds(ignore_no_image=True)
        if args.text_embeds:
            search_title_using_embeds(args.image, worlds, args.text_embeds)
        else:
            search_title(args.image, worlds)