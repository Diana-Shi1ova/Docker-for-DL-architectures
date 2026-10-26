from __future__ import annotations

import copy
import io
import os
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

app = FastAPI(title="ResNet18 Inference API", version="3.0")

CUDA_AVAILABLE = torch.cuda.is_available()
DEFAULT_DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"

BUTTERFLY_CHECKPOINT = Path(
    os.getenv("BUTTERFLY_CHECKPOINT", "/models/butterfly_resnet18.pth")
)

# Universal ImageNet model: create a CPU copy first.
IMAGENET_WEIGHTS = ResNet18_Weights.DEFAULT

imagenet_cpu = models.resnet18(
    weights=IMAGENET_WEIGHTS
).cpu().eval()

imagenet_models = {"cpu": imagenet_cpu}

# Create an independent GPU copy when CUDA is available.
if CUDA_AVAILABLE:
    imagenet_models["cuda"] = (
        copy.deepcopy(imagenet_cpu).to("cuda").eval()
    )

imagenet_transform = IMAGENET_WEIGHTS.transforms()
imagenet_categories = IMAGENET_WEIGHTS.meta["categories"]

# Butterfly model.
butterfly_models = {}
butterfly_transform = None
butterfly_classes = []
butterfly_checkpoint_error = None

BUTTERFLY_LABELS = {
    "admiral": "Red Admiral (Vanessa atalanta)",
    "painted_lady": "Painted Lady (Vanessa cardui)",
}


def load_butterfly_model():
    global butterfly_models
    global butterfly_transform
    global butterfly_classes
    global butterfly_checkpoint_error

    if not BUTTERFLY_CHECKPOINT.is_file():
        butterfly_checkpoint_error = (
            f"Checkpoint not found: {BUTTERFLY_CHECKPOINT}"
        )
        print(butterfly_checkpoint_error, flush=True)
        return

    try:
        checkpoint: dict[str, Any] = torch.load(
            BUTTERFLY_CHECKPOINT,
            map_location="cpu",
            weights_only=False,
        )

        butterfly_classes = list(checkpoint["classes"])

        if len(butterfly_classes) != 2:
            raise ValueError(
                f"Expected 2 butterfly classes, got {butterfly_classes}"
            )

        model = models.resnet18(weights=None)
        model.fc = nn.Linear(
            model.fc.in_features,
            len(butterfly_classes),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model = model.cpu().eval()

        butterfly_models = {"cpu": model}

        if CUDA_AVAILABLE:
            butterfly_models["cuda"] = (
                copy.deepcopy(model).to("cuda").eval()
            )

        image_size = int(checkpoint.get("image_size", 160))
        mean = tuple(
            checkpoint.get(
                "normalization_mean",
                (0.485, 0.456, 0.406),
            )
        )
        std = tuple(
            checkpoint.get(
                "normalization_std",
                (0.229, 0.224, 0.225),
            )
        )

        butterfly_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        print(
            "Butterfly model loaded. "
            f"Classes: {butterfly_classes}. "
            f"Available devices: {list(butterfly_models)}",
            flush=True,
        )

    except Exception as exc:
        butterfly_models = {}
        butterfly_transform = None
        butterfly_classes = []
        butterfly_checkpoint_error = str(exc)
        print(
            f"Failed to load butterfly model: {exc}",
            flush=True,
        )


load_butterfly_model()


def get_device_label(device: str) -> str:
    if device == "cuda":
        return f"GPU — {torch.cuda.get_device_name(0)} (CUDA)"
    return "CPU"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "default_device": DEFAULT_DEVICE,
        "available_devices": (
            ["cpu", "cuda"] if CUDA_AVAILABLE else ["cpu"]
        ),
        "gpu": (
            torch.cuda.get_device_name(0)
            if CUDA_AVAILABLE
            else None
        ),
        "models": {
            "imagenet": sorted(imagenet_models.keys()),
            "butterfly": sorted(butterfly_models.keys()),
        },
        "butterfly_model_error": butterfly_checkpoint_error,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Form(default="imagenet"),
    device: str = Form(default="auto"),
):
    model = model.strip().lower()
    if model not in {"imagenet", "butterfly"}:
        raise HTTPException(
            status_code=400,
            detail="model must be 'imagenet' or 'butterfly'",
        )

    device = device.strip().lower()
    if device == "auto":
        device = DEFAULT_DEVICE

    if device not in {"cpu", "cuda"}:
        raise HTTPException(
            status_code=400,
            detail="device must be 'cpu', 'cuda', or 'auto'",
        )

    if device == "cuda" and not CUDA_AVAILABLE:
        raise HTTPException(
            status_code=400,
            detail=(
                "GPU inference was requested, but CUDA is not "
                "available in the inference container."
            ),
        )

    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty",
            )

        image = Image.open(io.BytesIO(raw)).convert("RGB")

    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image",
        ) from exc
    finally:
        await file.close()

    # Specialized two-class butterfly model.
    if model == "butterfly":
        if device not in butterfly_models or butterfly_transform is None:
            detail = (
                "Butterfly model is unavailable. "
                f"Checkpoint: {BUTTERFLY_CHECKPOINT}. "
                f"Details: {butterfly_checkpoint_error or 'unknown error'}"
            )
            raise HTTPException(status_code=503, detail=detail)

        tensor = butterfly_transform(image).unsqueeze(0).to(device)

        with torch.inference_mode():
            probabilities = torch.softmax(
                butterfly_models[device](tensor),
                dim=1,
            )[0]

        ranked = torch.argsort(
            probabilities,
            descending=True,
        ).tolist()

        predictions = []
        for idx in ranked:
            class_name = butterfly_classes[idx]
            predictions.append({
                "label": BUTTERFLY_LABELS.get(
                    class_name,
                    class_name,
                ),
                "class_name": class_name,
                "confidence": float(probabilities[idx].item()),
            })

        return {
            "model": "butterfly",
            "model_name": "Butterfly-specific model",
            "device": get_device_label(device),
            "device_id": device,
            "predictions": predictions,
        }

    # Universal ImageNet model.
    tensor = imagenet_transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        probabilities = torch.softmax(
            imagenet_models[device](tensor),
            dim=1,
        )[0]

    values, indices = torch.topk(probabilities, k=5)

    predictions = [
        {
            "label": imagenet_categories[int(idx)],
            "confidence": float(score),
        }
        for score, idx in zip(
            values.tolist(),
            indices.tolist(),
        )
    ]

    return {
        "model": "imagenet",
        "model_name": "Universal ResNet18 (ImageNet, 1,000 classes)",
        "device": get_device_label(device),
        "device_id": device,
        "predictions": predictions,
    }