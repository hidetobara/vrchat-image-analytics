import argparse
import time
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision
from transformers import AutoProcessor, CLIPModel, AutoTokenizer
import numpy
import random

import util

MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 20
TRAIN_DIM = 128
MODEL_TRAINED = "/app/models/vrchat-worlds"
MODEL_PROCESSOR = "/app/models/processor"

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
    
    def append(self, text, image):
        self.texts.append(text)
        self.images.append(image)

    def load_dataset(self, path, limit, dl_dir="/app/data/images"):
        worlds = util.load_worlds(path)
        for w in worlds:
            img_path = os.path.join(dl_dir, w["id"] + ".png")
            if not os.path.exists(img_path):
                continue
            title = w["author"] + " " + w["title"]
            img = util.load_good_image(img_path, resize=256)
            self.append(title, numpy.array(img))
            if len(self.texts) >= limit:
                break
        print("LOADED_WORLDS=", len(self.texts))

    def divide(self, picked, mod=3):
        stride = len(self) // picked
        mod = mod % stride
        train = TitleAndImage()
        validation = TitleAndImage()
        for i in range(0, len(self)):
            txt, img = self[i]
            if i % stride == mod:
                validation.append(txt, img)
            else:
                train.append(txt, img)
        print("DIVIDED=", len(train), len(validation))
        return train, validation

def train(dataset_path="/app/data/best_worlds.csv", limit=100000):
    dataset = TitleAndImage()
    dataset.load_dataset(dataset_path, limit)
    train_data, validation_data = dataset.divide(100)
    train_loader = DataLoader(train_data, batch_size=TRAIN_DIM, shuffle=True, num_workers=1, drop_last=True)
    VALIDATION_DIM = len(validation_data)
    validation_loader = DataLoader(validation_data, batch_size=VALIDATION_DIM, num_workers=1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, eps=1e-6, weight_decay=0.2)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    loss_img_and_txt = nn.CrossEntropyLoss(reduction="mean")

    start = time.time()
    for epoch in range(EPOCHS):
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            texts, images = batch
            cropped_images = []
            for image in images:
                width, height, _ = image.shape
                wp = random.randint(0, width-224)
                hp = random.randint(0, height-224)
                cropped = image[wp:wp+224, hp:hp+224, :]
                cropped_images.append(cropped)
            cropped_images = torch.stack(cropped_images, dim=0)

            texts = tokenizer(texts, padding=True, truncation=True, max_length=32, return_tensors="pt")
            images = processor(images=cropped_images, return_tensors="pt")
            texts = texts.to(DEVICE)
            images = images.to(DEVICE)

            # Forward pass
            outputs = model(**texts, **images)
            #print("OUTPUTS=", outputs)

            # Compute loss
            #print("TXT=", outputs.logits_per_text.shape)
            #print("IMG=", outputs.logits_per_image)
            ground_truth = torch.arange(TRAIN_DIM, dtype=torch.long, device=DEVICE)
            loss = loss_img_and_txt(outputs.logits_per_text, ground_truth) + loss_img_and_txt(outputs.logits_per_image, ground_truth)

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            print(f"Loss: {loss.item():.4f}\r", end="")
            train_loss += loss.item()

        for batch in validation_loader:
            texts, images = batch
            texts = tokenizer(texts, padding=True, truncation=True, max_length=32, return_tensors="pt")
            images = processor(images=images, return_tensors="pt")
            texts = texts.to(DEVICE)
            images = images.to(DEVICE)
            # Forward pass
            outputs = model(**texts, **images)
            ground_truth = torch.arange(VALIDATION_DIM, dtype=torch.long, device=DEVICE)
            validation_loss = loss_img_and_txt(outputs.logits_per_text, ground_truth) + loss_img_and_txt(outputs.logits_per_image, ground_truth)

        passed = time.time() - start
        print(f"\n{passed:.1f} sec, Epoch {epoch}/{EPOCHS}, Train Loss: {train_loss:.4f}, Validation Loss {validation_loss.item():.4f}\n")
        model.save_pretrained(MODEL_TRAINED)
        processor.save_pretrained(MODEL_PROCESSOR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Let's party !")
    parser.add_argument('--train', action="store_true", help="train")
    parser.add_argument('--limit', type=int, default=100000, help="limit")
    args = parser.parse_args()

    if args.train:
        train(limit=args.limit)