import argparse
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel, AutoTokenizer

import util

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def search_title(image_paths, worlds, model_path, processor_path):
    model = AutoModel.from_pretrained(model_path).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(processor_path)
    processor = AutoProcessor.from_pretrained(processor_path)

    titles = list(map(lambda x: x["author"] + " " + x["title"], worlds))
    text_inputs = tokenizer(titles, padding="max_length", truncation=True, max_length=64, return_tensors="pt").to(DEVICE)

    for image_path in image_paths:
        img = util.load_good_image(image_path)
        print("IMAGE_PATH=", image_path)

        with torch.no_grad():
            image_inputs = processor(images=img, return_tensors="pt").to(DEVICE)
            outputs = model(**text_inputs, **image_inputs)
            logits_per_image = outputs.logits_per_image
            text_embeds = outputs.text_embeds
            probs = logits_per_image.softmax(dim=1).cpu()
        probs = torch.reshape(probs, [-1])
        top_values, top_indices = torch.topk(probs, k=5)
        for r, idx in enumerate(top_indices):
            print("P=", top_values[r], "Title=", titles[idx])
        torch.save(text_embeds, "/app/data/worlds.text.pt")
        break


def search_title_using_embeds(image_paths, worlds, embeds_path, model_path, processor_path):
    model = AutoModel.from_pretrained(model_path).to(DEVICE)
    processor = AutoProcessor.from_pretrained(processor_path)

    text_embeds = torch.load(embeds_path).cpu()

    for image_path in image_paths:
        img = util.load_good_image(image_path)
        print("IMAGE_PATH=", image_path)

        with torch.no_grad():
            inputs = processor(images=img, return_tensors="pt").to(DEVICE)
            vision_outputs = model.vision_model(**inputs)
            image_embeds = vision_outputs.pooler_output
            image_embeds = (image_embeds / image_embeds.norm(dim=-1, keepdim=True)).cpu()
        dist = torch.cdist(text_embeds, image_embeds)
        dist = torch.reshape(dist, [-1])
        indices = torch.argsort(dist)
        for r in range(5):
            idx = indices[r]
            print("DIST=", dist[idx], "IDX=", idx, "TITLE=", worlds[idx])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Let's evaluate !")
    parser.add_argument('--image', nargs='+', help="image path(s) for evaluate")
    parser.add_argument('--text_embeds', help="using text embeds")
    parser.add_argument('--model', default="/app/tuned/tmp", help="path to fine-tuned model")
    parser.add_argument('--processor', default="/app/tuned/tmp_processor", help="path to processor")
    args = parser.parse_args()

    if args.image:
        worlds = util.load_worlds(ignore_no_image=True)
        if args.text_embeds:
            search_title_using_embeds(args.image, worlds, args.text_embeds, args.model, args.processor)
        else:
            search_title(args.image, worlds, args.model, args.processor)
