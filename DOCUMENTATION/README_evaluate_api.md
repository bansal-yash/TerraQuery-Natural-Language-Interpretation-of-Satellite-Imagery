# Evaluate API

## General Purpose
Comprehensive evaluation suite for GeoNLI Visual Question Answering system. Computes BERT-BLEU, grounding metrics with IoU matching, binary/numeric accuracy, and generates annotated visualizations with ground truth and predicted bounding boxes.

## Files

### `evaluate.py`
Main evaluation script implementing BERT-BLEU semantic n-gram precision, grounding metric with Hungarian matching and Count Penalty, binary exact match, and numeric exponential error scoring. Generates annotated images with GT (green) and predicted (red) bounding boxes.

### `inference.py`
Runs inference on evaluation datasets by making API requests to the VQA system. Handles classification of answer types (semantic/binary/numeric) and converts oriented bounding boxes to flat coordinate lists.

### `metrics_per_file.json`
Detailed per-file performance metrics including BERT-BLEU scores, grounding IoU, and task-specific accuracies for each evaluation sample.

### `eval_metrics.json`
Aggregated evaluation metrics summary providing overall system performance across all test samples.

### `out.json`
Inference output file containing model predictions with bounding boxes and answers for comparison with ground truth.
