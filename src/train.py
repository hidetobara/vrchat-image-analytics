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
EPOCHS = 10
CLIP_DIM = 64
MODEL_TRAINED = "/app/models/vrchat"

model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
processor = AutoProcessor.from_pretrained(MODEL_NAME)


class TitleAndImage(Dataset):
    def __init__(self):
        self.texts = []
        self.images = []

    def __getitem__(self, index) -> list:
        return self.texts[index], self.images[index]

    def __len__(self) -> int:
        return len(self.images)

    def load_dataset(self, path, dl_dir="/app/data/images"):
        with open(path, "r") as f:
            for items in csv.reader(f):
                if len(items) < 4 or items[0].startswith("#"):
                    continue
                wid = items[0]
                title = items[1]
                img_path = os.path.join(dl_dir, wid + ".png")
                if not os.path.exists(img_path):
                    continue
                self.texts.append(title)
                img = Image.open(img_path)
                img = img.convert("RGB")
                img = img.resize((256, 256), Image.Resampling.LANCZOS)
                self.images.append(numpy.array(img))

def train(dataset_path="/app/data/best_worlds.csv"):
    dataset = TitleAndImage()
    dataset.load_dataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=1, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, eps=1e-6, weight_decay=0.2)
    loss_img_and_txt = nn.CrossEntropyLoss(reduction="mean") # nn.MSELoss(reduction="sum")

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()

            texts, images = batch
            texts = tokenizer(texts, padding=True, truncation=True, max_length=32, return_tensors="pt")
            images = processor(images=images, return_tensors="pt")

            texts = texts.to(DEVICE)
            images = images.to(DEVICE)

            # Forward pass
            outputs = model(**texts, **images)
            #print("OUTPUTS=", outputs)

            # Compute loss
            #print("TXT=", outputs.logits_per_text.shape)
            #print("IMG=", outputs.logits_per_image)
            ground_truth = torch.arange(CLIP_DIM, dtype=torch.long, device=DEVICE)
            loss = loss_img_and_txt(outputs.logits_per_text, ground_truth) + loss_img_and_txt(outputs.logits_per_image, ground_truth)

            # Backward pass
            loss.backward()
            optimizer.step()
            print(f"Loss: {loss.item():.4f}\r", end="")
            total_loss += loss.item()

        print(f"\nEpoch {epoch}/{EPOCHS}, Total Loss: {total_loss:.4f}\n")
        model.save_pretrained(MODEL_TRAINED)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Let's party !")
    parser.add_argument('--train', action="store_true", help="train")
    args = parser.parse_args()

    if args.train:
        train()