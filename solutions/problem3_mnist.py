"""Problem 3: LIME and SHAP on MNIST using a CNN model."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lime import lime_image
from skimage.segmentation import mark_boundaries
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from solutions.utils.reporting import dump_json, ensure_dir, write_markdown


plt.switch_backend("Agg")


RANDOM_STATE = 42
REPORT_DIR = Path("reports/problem3")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 3


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def _set_seed(seed: int = RANDOM_STATE) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _prepare_data() -> Tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader


def _train(model: nn.Module, train_loader: DataLoader) -> None:
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{EPOCHS} - loss: {epoch_loss:.4f}")


def _evaluate(model: nn.Module, test_loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / total


def _lime_explanations(model: nn.Module, images: torch.Tensor, output_dir: Path) -> List[str]:
    explainer = lime_image.LimeImageExplainer(random_state=RANDOM_STATE)

    def predict_fn(batch: np.ndarray) -> np.ndarray:
        if batch.shape[-1] == 3:
            batch = batch[..., [0]]  # use first channel
        tensors = torch.from_numpy(batch.transpose((0, 3, 1, 2))).float()
        tensors = tensors.to(DEVICE)
        with torch.no_grad():
            logits = model(tensors)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    lime_summaries = []
    for i, image in enumerate(images):
        np_image = image.squeeze(0).cpu().numpy()
        np_image = np.expand_dims(np_image, axis=2)
        rgb_image = np.repeat(np_image, 3, axis=2)
        explanation = explainer.explain_instance(
            rgb_image,
            classifier_fn=predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=1000,
        )
        top_label = explanation.top_labels[0]
        temp, mask = explanation.get_image_and_mask(
            label=top_label,
            positive_only=False,
            num_features=6,
            hide_rest=False,
        )
        temp_norm = temp / (np.max(np.abs(temp)) + 1e-8)
        highlighted = mark_boundaries(temp_norm, mask)
        path = output_dir / f"lime_sample_{i}.png"
        plt.figure(figsize=(3, 3))
        plt.imshow(highlighted, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

        contributions = explanation.local_exp[top_label]
        contributions = sorted(contributions, key=lambda item: abs(item[1]), reverse=True)[:5]
        summary = ", ".join(f"feature {idx} ({weight:+.3f})" for idx, weight in contributions)
        lime_summaries.append(f"- Sample {i}: label {top_label}; {summary}; figure: {path}")
    return lime_summaries


def _shap_explanations(model: nn.Module, background: torch.Tensor, images: torch.Tensor, output_dir: Path) -> List[str]:
    model.eval()
    background = background.to(DEVICE)
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(images.to(DEVICE), check_additivity=False)

    shap_summaries = []
    def _ensure_chw(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            return np.expand_dims(arr, axis=0)
        if arr.ndim == 3:
            if arr.shape[0] in (1, 3):
                return arr
            if arr.shape[-1] in (1, 3):
                return np.moveaxis(arr, -1, 0)
        return arr

    for i in range(images.size(0)):
        image = images[i].cpu().numpy().squeeze(0)
        predicted = int(model(images[i:i + 1].to(DEVICE)).argmax(dim=1).item())

        if isinstance(shap_values, (list, tuple)):
            shap_contrib = [_ensure_chw(sv[i]) for sv in shap_values]
            pixel_importance = np.mean(np.abs(shap_values[predicted][i]))
        else:
            sample_values = shap_values[i]
            shap_contrib = [_ensure_chw(sample_values[class_idx]) for class_idx in range(sample_values.shape[0])]
            pixel_importance = np.mean(np.abs(sample_values[predicted]))

        path = output_dir / f"shap_sample_{i}.png"
        shap.image_plot(shap_contrib, np.expand_dims(image, axis=0), show=False)
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        shap_summaries.append(
            f"- Sample {i}: predicted {predicted}; mean |SHAP|={pixel_importance:.4f}; figure: {path}"
        )
    return shap_summaries


def run(output_dir: Path | str = REPORT_DIR) -> Dict[str, object]:
    output_dir = ensure_dir(output_dir)
    _set_seed()

    train_loader, test_loader = _prepare_data()
    model = SimpleCNN().to(DEVICE)
    _train(model, train_loader)
    accuracy = _evaluate(model, test_loader)

    # Select 5 samples from the test set
    test_iter = iter(test_loader)
    test_images, test_labels = next(test_iter)
    images = test_images[:5]
    images = images.to(DEVICE)

    lime_dir = ensure_dir(output_dir / "lime")
    shap_dir = ensure_dir(output_dir / "shap")

    lime_summaries = _lime_explanations(model, images.cpu(), lime_dir)

    background_batch, _ = next(iter(train_loader))
    background = background_batch[:50]
    shap_summaries = _shap_explanations(model, background, images, shap_dir)

    write_markdown(
        output_dir / "summary.md",
        [
            "# Problem 3 – LIME and SHAP on MNIST",
            "",
            f"Test accuracy of the CNN: **{accuracy:.3%}**.",
            "",
            "## LIME Explanations",
            "",
            *lime_summaries,
            "",
            "## SHAP Explanations",
            "",
            *shap_summaries,
        ],
    )

    dump_json(
        output_dir / "metadata.json",
        {
            "accuracy": accuracy,
            "device": str(DEVICE),
            "lime_images": [str((output_dir / "lime" / f"lime_sample_{i}.png")) for i in range(5)],
            "shap_images": [str((output_dir / "shap" / f"shap_sample_{i}.png")) for i in range(5)],
        },
    )

    return {"accuracy": accuracy}


if __name__ == "__main__":
    run()
