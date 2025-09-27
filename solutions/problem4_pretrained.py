"""Problem 4: LIME and SHAP with a pretrained ResNet50 on CIFAR-10."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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


def _extract_normalization(preprocess, weights) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean and std tensors used for normalization.

    TorchVision has evolved the way pretrained weight presets expose their
    preprocessing pipelines.  Some presets are simple ``Compose`` objects with
    an explicit :class:`~torchvision.transforms.Normalize`, others store the
    statistics directly on the preset object, and older releases relied on the
    ``meta`` dictionary.  The original implementation only checked for
    ``transforms.Normalize`` instances, which breaks with the newer
    ``ImageClassification`` presets that skip composing the individual
    transforms.  This helper now probes each of those locations so that we can
    always recover the correct statistics.
    """

    def _tensorise_stats(mean, std) -> tuple[torch.Tensor, torch.Tensor] | None:
        if mean is None or std is None:
            return None
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device=DEVICE)
        std_tensor = torch.as_tensor(std, dtype=torch.float32, device=DEVICE)
        if mean_tensor.ndim == 1:
            mean_tensor = mean_tensor.view(-1, 1, 1)
        if std_tensor.ndim == 1:
            std_tensor = std_tensor.view(-1, 1, 1)
        return mean_tensor, std_tensor

    # 1) The preset itself may expose ``mean`` and ``std`` attributes (newer
    # torchvision versions use ``ImageClassification`` objects that behave this
    # way).
    stats = _tensorise_stats(getattr(preprocess, "mean", None), getattr(preprocess, "std", None))
    if stats is not None:
        return stats

    # 2) Fall back to inspecting composed transforms (covers the classic
    # ``transforms.Compose`` case).
    transforms_seq = getattr(preprocess, "transforms", None)
    if transforms_seq is not None:
        for transform in transforms_seq:
            stats = _tensorise_stats(getattr(transform, "mean", None), getattr(transform, "std", None))
            if stats is not None:
                return stats

    # ``torchvision.transforms.v2`` pipelines are ``nn.Module`` instances.  If
    # available, iterating over ``children`` helps cover those cases without
    # importing the v2 API directly.
    if hasattr(preprocess, "children"):
        for transform in preprocess.children():
            stats = _tensorise_stats(getattr(transform, "mean", None), getattr(transform, "std", None))
            if stats is not None:
                return stats

    # 3) Finally consult the metadata dictionary as a last resort (older
    # torchvision releases).
    meta = getattr(weights, "meta", {}) or {}
    stats = _tensorise_stats(meta.get("mean"), meta.get("std"))
    if stats is not None:
        return stats

    raise ValueError("Unable to determine normalization statistics from weights.")


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
    shap_result = explainer.shap_values(batch)

    shap_indexes: np.ndarray | None = None
    shap_values: Sequence[torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray
    if isinstance(shap_result, tuple) and len(shap_result) == 2:
        shap_values, shap_indexes = shap_result
        shap_indexes = np.asarray(shap_indexes)
    else:
        shap_values = shap_result

    if not isinstance(shap_values, (list, tuple)):
        shap_values = [shap_values]

    def _to_numpy(value) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _ensure_hwc(arr: np.ndarray) -> np.ndarray:
        """Normalise SHAP outputs to ``(N, H, W, C)`` regardless of backend."""

        if arr.ndim == 5:
            # ``(N, K, C, H, W)`` – split the ranked axis outside this helper.
            raise ValueError("Ranked SHAP values should be split before calling _ensure_hwc.")

        if arr.ndim == 4:
            # Channel-first ``(N, C, H, W)``
            if arr.shape[1] <= 4 and arr.shape[1] != arr.shape[-1]:
                return np.transpose(arr, (0, 2, 3, 1))
            # Already channel-last ``(N, H, W, C)``
            if arr.shape[-1] <= 4:
                return arr
            # Fall back to treating the second dimension as channels.
            return np.transpose(arr, (0, 2, 3, 1))

        if arr.ndim == 3:
            # Add the batch axis back and recurse.
            return _ensure_hwc(arr[np.newaxis, ...])

        if arr.ndim == 2:
            # Single-channel heatmap without spatial channels.
            return arr[np.newaxis, :, :, np.newaxis]

        raise ValueError(f"Unsupported SHAP value shape: {arr.shape}")

    shap_values_np: List[np.ndarray] = []
    for sv in shap_values:
        arr = _to_numpy(sv)

        if arr.ndim == 5:
            # ``GradientExplainer`` can return ranked outputs as ``(N, K, C, H, W)``.
            ranked = np.transpose(arr, (1, 0, 2, 3, 4))
            for rank_arr in ranked:
                shap_values_np.append(_ensure_hwc(rank_arr))
            continue

        shap_values_np.append(_ensure_hwc(arr))

    mean, std = _extract_normalization(preprocess, weights)

    shap_summaries = []
    for idx in range(batch.size(0)):
        tensor_image = batch[idx]
        denorm = (tensor_image * std + mean).clamp(0, 1)
        np_image = denorm.cpu().numpy().transpose(1, 2, 0)

        shap_contrib = [sv[idx] for sv in shap_values_np]

        with torch.no_grad():
            logits = model(batch[idx: idx + 1])
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_idx = int(np.argmax(probs))

        sample_indexes: List[int] | None = None
        if shap_indexes is not None:
            sample_slice = np.asarray(shap_indexes[idx])
            if sample_slice.ndim == 0:
                sample_indexes = [int(sample_slice.item())]
            else:
                sample_indexes = [int(x) for x in sample_slice.tolist()]

        if sample_indexes is not None:
            try:
                target_pos = sample_indexes.index(predicted_idx)
                target_class = predicted_idx
            except ValueError:
                target_pos = 0
                target_class = sample_indexes[0]
        else:
            if predicted_idx < len(shap_contrib):
                target_pos = predicted_idx
                target_class = predicted_idx
            else:
                target_pos = int(np.argmax(probs[: len(shap_contrib)]))
                target_class = target_pos

        shap_map = shap_contrib[target_pos].sum(axis=2)
        vmax = np.max(np.abs(shap_map)) + 1e-8

        path = output_dir / f"shap_image_{idx}.png"
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(np_image)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(np_image)
        axes[1].imshow(shap_map, cmap="seismic", alpha=0.6, vmin=-vmax, vmax=vmax)
        axes[1].set_title(f"SHAP for {class_names[target_class]}")
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close(fig)

        shap_summaries.append(
            f"- Image {idx}: predicted {class_names[predicted_idx]} (p={probs[predicted_idx]:.3f}); "
            f"visualised class: {class_names[target_class]}; figure: {path}"
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
