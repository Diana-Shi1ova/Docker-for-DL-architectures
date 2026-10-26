#!/usr/bin/env python3
"""Fine-tune a pretrained ResNet18 to distinguish two butterfly species.

Expected mounted folders:
  /data/train/<class_name>/*
  /data/val/<class_name>/*
  /data/test/<class_name>/*

Writes the best validation checkpoint to /models/butterfly_resnet18.pth.
"""
from __future__ import annotations

import os
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/models"))
EPOCHS = int(os.getenv("EPOCHS", "5"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "160"))
SEED = int(os.getenv("SEED", "42"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "0"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(path: Path, transform, *, shuffle: bool) -> DataLoader:
    dataset = datasets.ImageFolder(path, transform=transform)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_function = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = loss_function(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    if total == 0:
        raise RuntimeError(f"No images found in {loader.dataset.root}")
    return total_loss / total, correct / total


def main() -> None:
    if EPOCHS < 1 or BATCH_SIZE < 1 or IMAGE_SIZE < 64:
        raise ValueError("EPOCHS/BATCH_SIZE must be positive and IMAGE_SIZE must be >= 64")
    for split in ("train", "val", "test"):
        if not (DATA_DIR / split).is_dir():
            raise FileNotFoundError(f"Missing {DATA_DIR / split}. Run prepare_data.py first.")

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB")
    else:
        print("WARNING: CUDA is unavailable; training will run on CPU.")
    print(f"Settings: epochs={EPOCHS}, batch_size={BATCH_SIZE}, image_size={IMAGE_SIZE}, seed={SEED}")

    # ImageNet normalization matches the pretrained ResNet18 weights.
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_loader = make_loader(DATA_DIR / "train", train_transform, shuffle=True)
    val_loader = make_loader(DATA_DIR / "val", eval_transform, shuffle=False)
    test_loader = make_loader(DATA_DIR / "test", eval_transform, shuffle=False)

    classes = train_loader.dataset.classes
    class_to_idx = train_loader.dataset.class_to_idx
    for name, loader in (("val", val_loader), ("test", test_loader)):
        if loader.dataset.class_to_idx != class_to_idx:
            raise ValueError(
                f"Class folders in {name} do not match train. "
                f"train={class_to_idx}, {name}={loader.dataset.class_to_idx}"
            )
    if len(classes) != 2:
        raise ValueError(f"Expected exactly 2 classes, found {classes}")

    print(f"Classes (label index): {class_to_idx}")
    for name, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
        counts = {c: 0 for c in classes}
        for _, label in loader.dataset.samples:
            counts[classes[label]] += 1
        print(f"{name}: {len(loader.dataset)} images; per class: {counts}")

    # Uses the same cached ImageNet pretrained weights as the inference service.
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    for parameter in model.parameters():
        parameter.requires_grad = False
    # Fine-tune the last feature block and the new two-class classifier.
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(device)

    optimizer = AdamW([
        {"params": model.layer4.parameters(), "lr": 1e-4},
        {"params": model.fc.parameters(), "lr": 5e-4},
    ], weight_decay=1e-4)
    loss_function = nn.CrossEntropyLoss()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / "butterfly_resnet18.pth"
    best_val_accuracy = -1.0
    started = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * labels.size(0)
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        val_loss, val_accuracy = evaluate(model, val_loader, device)
        train_loss = train_loss_sum / max(train_total, 1)
        train_accuracy = train_correct / max(train_total, 1)
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.3f}",
            flush=True,
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                "state_dict": model.state_dict(),
                "classes": classes,
                "class_to_idx": class_to_idx,
                "image_size": IMAGE_SIZE,
                "normalization_mean": mean,
                "normalization_std": std,
                "seed": SEED,
                "epoch": epoch,
                "best_val_accuracy": best_val_accuracy,
            }, checkpoint_path)
            print(f"Saved best checkpoint: {checkpoint_path}", flush=True)

    # Final test evaluation uses the checkpoint selected by validation accuracy.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_loss, test_accuracy = evaluate(model, test_loader, device)
    elapsed = time.time() - started
    print("\nTraining complete")
    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"Best validation accuracy: {checkpoint['best_val_accuracy']:.3f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.3f} ({test_accuracy * 100:.1f}%)")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Elapsed: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
