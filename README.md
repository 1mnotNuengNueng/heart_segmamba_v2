# heart_segmamba_v2

Dataset: https://drive.google.com/drive/folders/1AEbkkW-vcLik5GH-JnEHKIg_isNOtf8H?usp=drive_link
Pretrained weights: https://drive.google.com/drive/folders/1RG5xrvHh2_OucuCkuWlLTsR1yjnAAjMp?usp=sharing

This project performs semantic segmentation for cardiac MRI on the ACDC dataset. This work is based on the research from [SegMamba-V2: Long-Range Sequential Modeling Mamba for Medical Image Segmentation](https://ieeexplore.ieee.org/document/11084842) and includes additional experiments and evaluations based on [Deep Learning Techniques for Automatic MRI Cardiac Multi-Structures Segmentation and Diagnosis: Is the Problem Solved?](https://ieeexplore.ieee.org/document/8360453).

It includes the full workflow for:

- environment setup
- data preparation and preprocessing
- training `SegMamba` and `UNETR`
- evaluation
- inference on both dataset splits and single `.nii/.nii.gz` files

The core logic lives in `acdc_framework/`, while model-specific entry scripts are provided in `SegMamba/` and `UNETR/`.

## Project Structure

```text
Heart_Segmamba/
|- acdc_framework/         # shared framework for preprocess/train/evaluate/predict
|- SegMamba/               # entrypoints for SegMamba
|- UNETR/                  # entrypoints for UNETR
|- shared_acdc/            # helpers for reading and exporting ACDC data
|- demo/                   # single-file NIfTI inference from .pt weights
|- requirements.txt
```

## Installation

Work from the project root:

```powershell
cd c:\Users\thana\OneDrive\Desktop\Heart_Segmamba
```

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Notes:

- You need a working `PyTorch` installation before training or inference.
- If you want GPU acceleration, install the PyTorch build that matches your CUDA version.
- If `mamba-ssm` is difficult to install on your Windows machine, start with `UNETR` first and handle the `SegMamba` dependency afterward.

## Expected ACDC Dataset Layout

The project reads dataset paths from `acdc_framework/settings.py` and expects the ACDC dataset at:

```text
ACDC/ACDC/database/training
ACDC/ACDC/database/testing
```

After downloading and extracting ACDC, the structure should look like this:

```text
Heart_Segmamba/
|- ACDC/
|  |- ACDC/
|  |  |- database/
|  |  |  |- training/
|  |  |  |- testing/
```

Each `patientXXX/` directory should contain files such as:

- `Info.cfg`
- `patient001_frame01.nii.gz`
- `patient001_frame01_gt.nii.gz`

This pipeline only uses the `ED` and `ES` phases.

## 1. Preprocess the Data

The preprocessing scripts in `SegMamba` and `UNETR` both call the same shared pipeline:

```powershell
python SegMamba\1_preprocess_acdc.py
```

or

```powershell
python UNETR\1_preprocess_acdc.py
```

Main outputs are written to:

- `acdc_runs/shared_preprocessed/train`
- `acdc_runs/shared_preprocessed/val`
- `acdc_runs/shared_preprocessed/test`
- `acdc_runs/shared_preprocessed/splits.json`

Each case is saved as:

- a `.npz` data file
- a `.json` metadata file

Default preprocessing settings:

- target shape: `16 x 192 x 192` in `D, H, W`
- normalization: `zscore`
- validation ratio: `0.2`

The pipeline also exports the dataset in nnU-Net raw format to:

- `nnUNet_raw/Dataset701_ACDC`

If you want custom preprocessing settings, call the framework directly:

```powershell
python -m acdc_framework.preprocess --target-shape 16 192 192 --normalization zscore --val-ratio 0.2
```

Useful options:

- `--output-dir` change the output directory for preprocessed data
- `--normalization {zscore|minmax|none}`
- `--skip-nnunet-export` skip nnU-Net raw export

## 2. Train a Model

### Train SegMamba

```powershell
python SegMamba\2_train_segmamba.py
```

This script calls `train_model("segmamba", batch_size=1)` and uses these main defaults:

- epochs: `1000`
- learning rate: `1e-4`
- num workers: `8`
- early stopping patience: `50`

### Train UNETR

```powershell
python UNETR\2_train_unetr.py
```

This script calls `train_model("transformer")`.

### Train with Custom Arguments

If you want full control over the training parameters, call the shared framework directly:

```powershell
python -m acdc_framework.train --model segmamba --epochs 200 --batch-size 1 --lr 1e-4 --num-workers 4 --patience 30
```

Example for UNETR:

```powershell
python -m acdc_framework.train --model transformer --epochs 200 --batch-size 1 --lr 1e-4 --num-workers 4 --patience 30
```

Important options:

- `--model {cnn|transformer|segmamba}`
- `--epochs`
- `--batch-size`
- `--lr`
- `--num-workers`
- `--preprocessed-root`
- `--disable-augmentation`
- `--seed`
- `--resume-checkpoint`
- `--patience`

Training outputs are stored in:

- `acdc_runs/<model>/checkpoints/best.pt`
- `acdc_runs/<model>/checkpoints/best_checkpoint.pt`
- `acdc_runs/<model>/checkpoints/last.pt`
- `acdc_runs/<model>/checkpoints/last_checkpoint.pt`
- `acdc_runs/<model>/history.json`
- `acdc_runs/<model>/history.csv`
- `acdc_runs/<model>/training_curves.png`
- `acdc_runs/<model>/train_summary.json`

Here, `<model>` is either `segmamba` or `transformer`.

### Resume Training from a Checkpoint

```powershell
python -m acdc_framework.train --model segmamba --resume-checkpoint acdc_runs\segmamba\checkpoints\best_checkpoint.pt
```

## 3. Run Inference

### 3.1 Predict an Entire Preprocessed Split

For SegMamba:

```powershell
python SegMamba\3_predict_segmamba.py
```

For UNETR:

```powershell
python UNETR\3_predict_unetr.py
```

Both scripts use these defaults:

- split: `test`
- checkpoint: `best.pt`

Outputs are written to:

- `acdc_runs/segmamba/predictions/test`
- `acdc_runs/transformer/predictions/test`

For each case, the pipeline saves:

- a prediction mask as `.nii.gz`
- a metadata file as `.json`

### 3.2 Predict with a Custom Split or Checkpoint

```powershell
python -m acdc_framework.predict --model segmamba --split val --checkpoint best.pt
```

Example for UNETR:

```powershell
python -m acdc_framework.predict --model transformer --split test --checkpoint last.pt
```

Important options:

- `--model {cnn|transformer|segmamba}`
- `--split {train|val|test}`
- `--checkpoint`
- `--preprocessed-root`

## 4. Evaluate Predictions

After prediction, run evaluation as follows.

SegMamba:

```powershell
python SegMamba\4_evaluate_segmamba.py
```

UNETR:

```powershell
python UNETR\4_evaluate_unetr.py
```

Or call the framework directly:

```powershell
python -m acdc_framework.evaluate --model segmamba --split test
```

Outputs are saved to:

- `acdc_runs/<model>/metrics/test_metrics.json`

The report includes metrics such as:

- mean Dice
- mean HD95
- mean AP50
- mean AP50:95
- challenge metrics

## 5. Predict a Single `.nii/.nii.gz` File

If you already have a `.pt` checkpoint and want to run inference on a single MRI file, use the script in `demo/`.

Place files here:

- model weights in `demo/weights/`
- input images in `demo/data/`

Then run:

```powershell
python demo\predict_pt.py --model transformer --image demo\data\case001.nii.gz --weights demo\weights\best.pt
```

Or for SegMamba:

```powershell
python demo\predict_pt.py --model segmamba --image demo\data\case001.nii.gz --weights demo\weights\best.pt
```

Outputs are written to:

- `demo/output/<filename>_pred.nii.gz`
- `demo/output/<filename>_pred.json`

This script will:

- resize the input volume to the target shape
- normalize the image
- run the model
- resize the predicted mask back to the original image shape

Important options:

- `--model {cnn|transformer|segmamba}`
- `--image`
- `--weights`
- `--output`
- `--target-shape`
- `--normalization {zscore|minmax|none}`

## 6. Recommended Workflow

If you are starting from scratch, run the pipeline in this order.

For SegMamba:

```powershell
python SegMamba\1_preprocess_acdc.py
python SegMamba\2_train_segmamba.py
python SegMamba\3_predict_segmamba.py
python SegMamba\4_evaluate_segmamba.py
```

For UNETR:

```powershell
python UNETR\1_preprocess_acdc.py
python UNETR\2_train_unetr.py
python UNETR\3_predict_unetr.py
python UNETR\4_evaluate_unetr.py
```

## 7. Common Issues

### Dataset Not Found

Make sure the dataset is actually located at:

```text
ACDC/ACDC/database/training
ACDC/ACDC/database/testing
```

### Checkpoint Not Found During Prediction

Make sure these files exist:

- `acdc_runs/segmamba/checkpoints/best.pt`
- `acdc_runs/transformer/checkpoints/best.pt`

If they do not exist yet, train the model first or specify an existing file with `--checkpoint`.

### Using a Different Preprocessed Data Directory

Use the `--preprocessed-root` option during training and prediction.

Example:

```powershell
python -m acdc_framework.train --model transformer --preprocessed-root D:\datasets\acdc_preprocessed
python -m acdc_framework.predict --model transformer --preprocessed-root D:\datasets\acdc_preprocessed
```

## 8. Additional Notes

- `SegMamba/` and `UNETR/` provide quick entry scripts.
- For detailed parameter control, use `python -m acdc_framework.<module>`.
- The framework also supports a `cnn` model, even though it does not have separate shortcut scripts like `SegMamba` and `UNETR`.
