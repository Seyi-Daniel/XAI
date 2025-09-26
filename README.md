# Explainable AI Programming Problems

This repository contains fully scripted solutions for the four programming problems outlined in `problems.pdf`. Each pipeline trains the requested models, generates LIME and/or SHAP explanations, and saves human-readable reports under the `reports/` directory.

## Environment setup

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

The requirements include `scikit-learn`, `lime`, `shap`, `torch`, and `torchvision`. CPU execution is supported; GPU acceleration is used automatically if available.

## Usage

All tasks can be executed through the `main.py` CLI:

```bash
python main.py <problem> [--output-dir reports]
```

Where `<problem>` is one of:

- `problem1` – Diabetes classification with LIME analyses.
- `problem2` – Breast Cancer classification with SHAP.
- `problem3` – MNIST CNN with both LIME and SHAP image explanations.
- `problem4` – Pretrained ResNet50 fine-tuned on CIFAR-10 with LIME and SHAP.
- `all` – Run every pipeline sequentially.

Each command creates a dedicated subdirectory inside the output directory (defaults to `reports/`) containing:

- `summary.md` – narrative description of the results and explanation highlights.
- JSON files with structured metrics and metadata.
- PNG files for each generated explanation and visualisation.

### Example commands

Run only the first problem and inspect the outputs:

```bash
python main.py problem1
ls reports/problem1
cat reports/problem1/summary.md
```

Execute the complete workflow and gather all reports in a custom folder:

```bash
python main.py all --output-dir experiment_outputs
```

## Notes per problem

- **Problem 1**: Trains Logistic Regression, SVM, and MLP classifiers on the Diabetes dataset (converted to a binary task). Generates LIME explanations for five test samples and explores kernel width sensitivity.
- **Problem 2**: Repeats a three-model comparison on the Breast Cancer dataset. Uses SHAP (Kernel Explainer) to analyse five samples and generates summary and dependence plots.
- **Problem 3**: Trains a small CNN on MNIST for three epochs. Produces LIME and SHAP explanation figures for five test digits.
- **Problem 4**: Fine-tunes the final layer of a pretrained ResNet50 on a 10k-sample subset of CIFAR-10. Creates LIME and SHAP explanations for five test images, facilitating qualitative comparison.

The scripts print training progress to the console and save all intermediate artefacts for inclusion in the final report.
