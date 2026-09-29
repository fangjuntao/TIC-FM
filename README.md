# TIC-FM

This project evaluates a time-series classification model  **TIC-FM**. 

## Repository Structure

* `checkpoints/`: model weights and hyperparameter JSON files
* `scripts/`: evaluation scripts
* `src/`: core model and dataset loading code (TIC package)

## Main Script

* `scripts/eval_TSEncoder_orion_icl_classifier_ucr_full.py`

  * Supports two evaluation modes:

    * `direct`: in-context inference using a fixed support set
    * `classifier_v2`: uses `TSEncoderICLClassifierV2` (with class shift and `RandomCropResize` augmentation)
  * Supports both UCR and UEA dataset suites

## Environment

* Python 3.9+ (recommended to match the existing environment)
* PyTorch
* NumPy

> Dependencies follow the current environment. If you encounter missing packages, install them as prompted.

## Installation

We recommend creating an isolated conda environment:

```bash
conda create -n TICFS python=3.9 -y
conda activate TICFS
```

Install core dependencies (example):

```bash
pip install torch numpy
```

If you want to match an existing environment exactly, install missing dependencies according to the error messages.

Restore from an exported environment file:

```bash
conda env create -f environment.yml
conda activate TICFS
```

## Quick Start

Run from the project root:

```bash
python scripts/eval_TSEncoder_orion_icl_classifier_ucr_full.py \
  --suite ucr \
  --mode classifier_v2 \
  --ucr_path /path/to/UCRdata/ \
  --full_ckpt checkpoints/TSEncoder_orion_icl_full.pt \
  --model_hparams_json checkpoints/TSEncoder_orion_icl_model_hparams.json
```

Evaluate a single dataset:

```bash
python scripts/eval_TSEncoder_orion_icl_classifier_ucr_full.py \
  --suite ucr \
  --dataset ECG200 \
  --mode direct
```
## Benchmark Results

TIC-FM is evaluated in the [TSC-FM time series classification foundation model benchmark](https://tsc-fm.dmirlab.com/). See its [model configurations and benchmark results](https://tsc-fm.dmirlab.com/methods/tic-fm), compare matching settings on the [time series classification leaderboard](https://tsc-fm.dmirlab.com/leaderboard), and consult the [Standard and few-shot evaluation protocol](https://tsc-fm.dmirlab.com/evaluation).


## Common Arguments

* `--mode`: `direct` or `classifier_v2`
* `--suite`: `ucr` or `uea`
* `--dataset`: name of a specific dataset to evaluate
* `--support_size`: number of support samples for ICL
* `--query_batch_size`: query batch size
* `--softmax_temperature`: softmax temperature
* `--uea_fusion`: `concat` or `sum_embed`
* `--uea_use_var_selector`: enable variance-based channel selection

## Outputs

The script prints per-dataset accuracy and reports the overall average accuracy at the end.

## Notes
* Pretrained model parameters can be downloaded from the Hugging Face repository: https://huggingface.co/Jwpqkh/TIC-FM/tree/main.
* Model weights and hyperparameter JSON files are located in `checkpoints/` by default, and can be overridden via command-line arguments.
