import argparse
import time
import os
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from transformers import AutoProcessor, AutoModel, AutoTokenizer
import numpy
import random

import util

MODEL_NAME = "google/siglip2-base-patch16-256"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def siglip_loss(logits_per_image):
    """SigLIP sigmoid loss. Treats each pair independently as binary classification."""
    batch_size = logits_per_image.shape[0]
    labels = 2 * torch.eye(batch_size, device=logits_per_image.device) - 1  # 1 for match, -1 for non-match
    return -torch.mean(F.logsigmoid(labels * logits_per_image))


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

    def load_dataset(self, path, limit, dl_dir="/app/data/old/images"):
        worlds = util.load_worlds(path)
        for w in worlds:
            _, img_path = util.get_box_path(w["id"])
            if not os.path.exists(img_path):
                continue
            title = w["author"] + " " + w["title"]
            img = util.load_good_image(img_path, resize=256)
            self.append(title, numpy.array(img))
            if len(self.texts) >= limit:
                break
        print(img_path)
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


def train(model_path, batch_size, epochs, dataset_path="/app/data/best_worlds.csv", limit=100000):
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    processor_path = model_path + "_processor"

    dataset = TitleAndImage()
    dataset.load_dataset(dataset_path, limit)
    train_data, validation_data = dataset.divide(100)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
    VALIDATION_DIM = len(validation_data)
    validation_loader = DataLoader(validation_data, batch_size=VALIDATION_DIM, num_workers=1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, eps=1e-6, weight_decay=0.2)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    start = time.time()
    for epoch in range(epochs):
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            texts, images = batch
            cropped_images = []
            for image in images:
                width, height, _ = image.shape
                wp = random.randint(0, width - 224)
                hp = random.randint(0, height - 224)
                cropped = image[wp:wp+224, hp:hp+224, :]
                cropped_images.append(cropped)
            cropped_images = torch.stack(cropped_images, dim=0)

            texts = tokenizer(texts, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
            images = processor(images=cropped_images, return_tensors="pt")
            texts = texts.to(DEVICE)
            images = images.to(DEVICE)

            outputs = model(**texts, **images)
            loss = siglip_loss(outputs.logits_per_image)

            loss.backward()
            optimizer.step()
            scheduler.step()
            print(f"Loss: {loss.item():.4f}\r", end="")
            train_loss += loss.item()

        with torch.no_grad():
            for batch in validation_loader:
                texts, images = batch
                texts = tokenizer(texts, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
                images = processor(images=images, return_tensors="pt")
                texts = texts.to(DEVICE)
                images = images.to(DEVICE)
                outputs = model(**texts, **images)
                validation_loss = siglip_loss(outputs.logits_per_image)

        passed = time.time() - start
        print(f"\n{passed:.1f} sec, Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss {validation_loss.item():.4f}\n")
        model.save_pretrained(model_path)
        processor.save_pretrained(processor_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SigLIP fine-tuning for VRChat worlds")
    parser.add_argument('--train', action="store_true", help="train")
    parser.add_argument('--limit', type=int, default=1000, help="max training samples")
    parser.add_argument('--model', type=str, default="/app/tuned/tmp", help="path to save the fine-tuned model")
    parser.add_argument('--batch_size', type=int, default=128, help="training batch size")
    parser.add_argument('--epochs', type=int, default=10, help="number of training epochs")
    args = parser.parse_args()

    if args.train:
        train(limit=args.limit, model_path=args.model, batch_size=args.batch_size, epochs=args.epochs)
