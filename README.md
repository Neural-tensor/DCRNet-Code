# DCRNet

Official implementation for the paper:

**DCRNet: Dual-Stream Cross-Modal Information-Guided Feature Reconstruction for Multimodal Anomaly Detection**

The code trains and evaluates the dual-stream cross-modal feature reconstruction network on MVTec 3D-AD using RGB images and pre-extracted point-cloud features.

## Files

- `main.py`: training and evaluation entry point.
- `DCRNet.py`: DCRNet reconstruction network.
- `de_transformer.py`: cross-modal attention with neighborhood-aware masking and bottleneck design.
- `gen_attn_mask.py`: visualizes the 2D Gaussian distance-prior mask.
- `dataset.py`: dataset loading utilities.
- `resnet.py`: pretrained RGB feature extractor.
- `utils.py`: losses, metrics, visualization, and utility functions.

## Point-Cloud Features

Point-cloud features are extracted following M3DM, *Multimodal Industrial Anomaly Detection via Hybrid Fusion*. Use the M3DM feature extraction code to prepare point-cloud features for both the training and test splits of MVTec 3D-AD.

Example point-cloud feature paths:

- `.../M3DM-main/datasets/train`
- `.../M3DM-main/datasets/test`

## Configuration

Before running the code, update the dataset paths in `main.py`:

- `DATASET_PATH`: RGB image path for MVTec 3D-AD.
- `PonitCloudFeat_PATH`: extracted point-cloud feature path.

Then run:

```bash
python main.py
```
