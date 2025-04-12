# InverTune

This project provides tools for analyzing, detecting, and mitigating backdoor attacks in CLIP models.

## Project Structure

```
├── config/            # Configuration files
│   └── config.yaml    # Main configuration
├── models/            # Model implementations
│   └── inversion.py # Trigger Inversion engineering implementations
├── utils/             # Utility functions
│   └── metrics.py     # Evaluation metrics
├── data/              # Data handling (not included)
├── Activation.py    # Analyze model activations
├── ActivationTuning.py      # Selective fine-tuning
├── TargetLabel.py           # Target label analysis
├── TestImagenet.py          # ImageNet evaluation
├── TestMScoco.py            # MS-COCO evaluation
└── TriggerInversion.py       # Trigger Inversion engineering
```

## Features

- **Backdoor Detection**: Detect potential backdoors in CLIP models
- **Trigger Inversion Engineering**: Reconstruct potential backdoor triggers
- **Activation Analysis**: Analyze model activations to identify affected components
- **Selective Fine-Tuning**: Mitigate backdoors while preserving model performance
- **Evaluation Tools**: Comprehensive evaluation on ImageNet and MS-COCO datasets

## Usage

All scripts support configuration files and command-line arguments.

### Environment Setup

To set up the environment for InverTune, use the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate InverTune
```

### Configuration

Edit `config/config.yaml` to set up your parameters:

```yaml
data:
  data_root: "data/ILSVRC2012_img_val/ILSVRC2012_img_val"
  batch_size: 64
  num_workers: 16
  target_class_id: "n07734744"
  target_word: "mushroom"
  target_label: 947
  num_batches: 20
  epochs: 10

model:
  model_path: "checkpoints/model.pt"
  clip_type: "RN50"  # RN50, RN101, ViT-B/32, ViT-B/16
  device: "cuda:0"

train:
  num_steps: 1000
  learning_rate: 0.01

optimizer:
  mask_weight: 0.01     
  infonce_weight: 5
  emd_weight: 0.5
  ssim_weight: 1

trigger:
  inv_trigger_path: "results/trigger.pth"
```

### Example Commands

1. **Analyze Target Labels**:
   ```
   python TargetLabel.py --config config/config.yaml
   ```

2. **Inversion Engineer Triggers**:
   ```
   python TriggerReversal.py --config config/config.yaml
   ```

3. **Perform Activation Fine-Tuning**:
   ```
   python ActivationTuning.py --config config/config.yaml
   ```

4. **Test on ImageNet**:
   ```
   python TestImagenet.py --config config/config.yaml
   ```

5. **Test on MS-COCO**:
   ```
   python TestMScoco.py --config config/config.yaml
   ```
