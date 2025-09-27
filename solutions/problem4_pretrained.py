"""Problem 4: LIME and SHAP with a pretrained ResNet50 on CIFAR-10."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
import torch.nn as nn
import torch.optim as optim
from lime import lime_image
from PIL import Image
from skimage.segmentation import mark_boundaries
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

from solutions.utils.reporting import dump_json, ensure_dir, write_markdown


plt.switch_backend("Agg")


RANDOM_STATE = 42
REPORT_DIR = Path("reports/problem4")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
FINE_TUNE_EPOCHS = 2
TRAIN_SUBSET = 10000  # limit training set size to keep runtime manageable
NUM_IMAGES = 5


def _set_seed(seed: int = RANDOM_STATE) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _prepare_dataloaders(weights) -> Dict[str, DataLoader]:
    preprocess = weights.transforms()
    train_dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=preprocess,
    )
    test_dataset = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=preprocess,
    )

    if TRAIN_SUBSET < len(train_dataset):
        generator = torch.Generator().manual_seed(RANDOM_STATE)
        indices = torch.randperm(len(train_dataset), generator=generator)[:TRAIN_SUBSET]
        train_dataset = Subset(train_dataset, indices.tolist())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return {"train": train_loader, "test": test_loader}


def _load_model() -> tuple[nn.Module, object]:
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    model = model.to(DEVICE)
    return model, weights


def _fine_tune(model: nn.Module, dataloaders: Dict[str, DataLoader]) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    model.train()
    for epoch in range(FINE_TUNE_EPOCHS):
        running_loss = 0.0
        for inputs, targets in dataloaders["train"]:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloaders["train"].dataset)
        print(f"Fine-tune epoch {epoch + 1}/{FINE_TUNE_EPOCHS} - loss: {epoch_loss:.4f}")


def _evaluate(model: nn.Module, dataloaders: Dict[str, DataLoader]) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloaders["test"]:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / total


def _load_original_images(weights) -> Tuple[List[Image.Image], List[str]]:
    raw_dataset = datasets.CIFAR10(root="data", train=False, download=True, transform=None)
    images = []
    for i in range(NUM_IMAGES):
        image, _ = raw_dataset[i]
        images.append(image)
    return images, list(raw_dataset.classes)


def _lime_for_images(
    model: nn.Module,
    weights,
    original_images: List[Image.Image],
    class_names: List[str],
    output_dir: Path,
) -> List[str]:
    explainer = lime_image.LimeImageExplainer(random_state=RANDOM_STATE)
    preprocess = weights.transforms()

    def predict_fn(batch: np.ndarray) -> np.ndarray:
        tensors = []
        for array in batch:
            img = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
            tensor = preprocess(img).to(DEVICE)
            tensors.append(tensor)
        inputs = torch.stack(tensors)
        with torch.no_grad():
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
        return probs

    lime_summaries = []
    for idx, image in enumerate(original_images):
        np_image = np.array(image)
        explanation = explainer.explain_instance(
            np_image,
            classifier_fn=predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=1000,
        )
        top_label = explanation.top_labels[0]
        temp, mask = explanation.get_image_and_mask(
            top_label,
            positive_only=False,
            num_features=6,
            hide_rest=False,
        )
        temp_norm = temp / (np.max(np.abs(temp)) + 1e-8)
        highlighted = mark_boundaries(temp_norm, mask)
        path = output_dir / f"lime_image_{idx}.png"
        plt.figure(figsize=(3, 3))
        plt.imshow(highlighted)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        predicted = predict_fn(np.expand_dims(np_image, axis=0))[0]
        predicted_idx = int(np.argmax(predicted))
        summary = (
            f"- Image {idx}: predicted {class_names[predicted_idx]} (p={predicted[predicted_idx]:.3f}); "
            f"top label in explanation: {class_names[top_label]}; figure: {path}"
        )
        lime_summaries.append(summary)
    return lime_summaries


def _shap_for_images(
    model: nn.Module,
    weights,
    original_images: List[Image.Image],
    class_names: List[str],
    dataloaders: Dict[str, DataLoader],
    output_dir: Path,
) -> List[str]:
    preprocess = weights.transforms()
    background_images = []
    for inputs, _ in dataloaders["train"]:
        background_images.append(inputs[:8])
        if len(background_images) * 8 >= 32:
            break
    background = torch.cat(background_images, dim=0).to(DEVICE)

    explainer = shap.GradientExplainer(model, background)

    processed_images = []
    for image in original_images:
        tensor = preprocess(image).to(DEVICE)
        processed_images.append(tensor)
    batch = torch.stack(processed_images)
    shap_values = explainer.shap_values(batch)

    shap_summaries = []
    for idx in range(batch.size(0)):
        np_image = batch[idx].cpu().numpy().transpose(1, 2, 0)
        shap_contrib = [sv[idx].transpose(1, 2, 0) for sv in shap_values]
        path = output_dir / f"shap_image_{idx}.png"
        shap.image_plot(shap_contrib, np.expand_dims(np_image, axis=0), show=False)
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        with torch.no_grad():
            logits = model(batch[idx: idx + 1])
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_idx = int(np.argmax(probs))
        shap_summaries.append(
            f"- Image {idx}: predicted {class_names[predicted_idx]} (p={probs[predicted_idx]:.3f}); figure: {path}"
        )
    return shap_summaries


def run(output_dir: Path | str = REPORT_DIR) -> Dict[str, object]:
    output_dir = ensure_dir(output_dir)
    _set_seed()

    model, weights = _load_model()
    dataloaders = _prepare_dataloaders(weights)
    _fine_tune(model, dataloaders)
    accuracy = _evaluate(model, dataloaders)

    original_images, class_names = _load_original_images(weights)

    lime_dir = ensure_dir(output_dir / "lime")
    shap_dir = ensure_dir(output_dir / "shap")

    lime_summaries = _lime_for_images(model, weights, original_images, class_names, lime_dir)
    shap_summaries = _shap_for_images(
        model,
        weights,
        original_images,
        class_names,
        dataloaders,
        shap_dir,
    )

    write_markdown(
        output_dir / "summary.md",
        [
            "# Problem 4 – LIME and SHAP with Pretrained ResNet50",
            "",
            f"Test accuracy after fine-tuning: **{accuracy:.3%}**.",
            "",
            "## LIME Image Explanations",
            "",
            *lime_summaries,
            "",
            "## SHAP Image Explanations",
            "",
            *shap_summaries,
            "",
            "Compare the visual patterns highlighted by LIME and SHAP to assess intuitiveness.",
        ],
    )

    dump_json(
        output_dir / "metadata.json",
        {
            "accuracy": accuracy,
            "device": str(DEVICE),
            "lime_images": [str(lime_dir / f"lime_image_{i}.png") for i in range(NUM_IMAGES)],
            "shap_images": [str(shap_dir / f"shap_image_{i}.png") for i in range(NUM_IMAGES)],
        },
    )

    return {"accuracy": accuracy}


if __name__ == "__main__":
    run()
