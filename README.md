# Superpixel Segmentation Evaluation Framework

This repository contains the code and results for evaluating various superpixel segmentation methods across different datasets.

## Directory Structure

```
SuperpixelSegmentation/
├── main.py                     # Entry point for running segmentation experiments
├── src/                        # Core implementation
│   ├── utils.py                # Helper functions (loading, metrics)
│   ├── viz.py                  # Visualization utilities
│   └── logger_config.py        # Logging setup
├── config/                     # Configuration
│   └── master_quickshift_params.json # Parameters for Quickshift method
├── files/                      # Data and Results
│   ├── Data/                   # Datasets (Input)
│   ├── Results/                # Experiment Results (Output)
│   ├── Visualizations/         # Generated visualizations
│   └── plots/                  # Generated plots
└── analysis/                   # Analysis scripts
    ├── calculate_bootstrap.py  # Calculates bootstrap stats
    └── plot_rank_stability.py  # Generates rank stability plots
```

## Installation & Setup

This project uses `uv` for dependency management.

1. **Install uv**: Follow instructions at [astral.sh/uv](https://github.com/astral-sh/uv).
2. **Install Dependencies**:
   ```bash
   uv sync
   ```

## Data Download

Please download the datasets and extract them into the `files/Data/` directory:
- [WoundSeg Dataset](https://huggingface.co/datasets/subbareddyoota/wseg_dataset)
- [Cwdb Dataset](https://github.com/recogna-lab/datasets/tree/master/ComplexWoundDB)
- [AZH Dataset](https://github.com/uwm-bigdata/wound-segmentation/tree/master/data/wound_dataset)

The expected structure is:
```
files/Data/
├── azh/
├── wsnet/
├── cwdb/
└── cicatrix/
```

## Usage

### 1. Running Experiments
Use `main.py` to run a specific method on a dataset.
```bash
uv run main.py --dataset azh --method slic --num_pixel_per_spixel 550 --save_files
```
Arguments:
- `--dataset`: `azh`, `wsnet`, `cwdb`, `cicatrix`
- `--method`: `edtrs`, `slic`, `felzenszwalb`, `quickshift`, `watershed`
- `--num_pixel_per_spixel`: Target superpixel count (e.g., 550)

### 2. Generating Bootstrap Statistics
After running experiments, generate bootstrapped means (10 subsamples of 90% data) for robust analysis.
```bash
cd analysis
uv run calculate_bootstrap.py
```
This creates `summary_bootstrap.json` in each result folder.

### 3. Plotting Rank Stability
Generate plots showing how method rankings change across different superpixel counts.
```bash
cd analysis
uv run plot_rank_stability.py
```
Outputs:
- Plots: `files/plots/`