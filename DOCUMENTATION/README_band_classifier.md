# Band Classifier

## General Purpose
This folder contains code for classifying satellite imagery into different band types (SAR, RGB, or false-color). Uses a ResNet50-based classifier to automatically detect image modality.

## Files

### `convert.py`
Converts TIFF satellite images to PNG/JPG formats with automatic 16-bit to 8-bit normalization for visualization and processing.

### `type_classifier.py`
CLI tool for satellite image type classification into SAR/RGB/false-color using ResNet50 with custom fully-connected layers. Includes GPU safety checks and fallback to CPU.

### `checkpoint/`
Directory containing saved model checkpoints for the band classifier.
