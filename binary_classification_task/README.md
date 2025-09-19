# Binary Classification Evaluator

A comprehensive binary classification evaluation tool that provides DCASE-compatible metrics for anomaly detection and general binary classification tasks.

## Features

- **DCASE-Compatible Metrics**: Implements exact DCASE Task 2 evaluation methodology
- **Flexible Input**: Supports both CSV files and direct array input
- **Comprehensive Metrics**: AUC, pAUC, precision, recall, F1, accuracy, and more
- **Statistical Confidence**: Jackknife resampling for 95% confidence intervals
- **Multiple Output Formats**: JSON, CSV, and console output
- **Official Score**: Harmonic mean calculation for balanced performance assessment

## Installation

### Requirements
```bash
pip install numpy pandas scikit-learn scipy
```

### Dependencies
- `numpy` - Numerical computations
- `pandas` - Data handling
- `scikit-learn` - Machine learning metrics
- `scipy` - Statistical functions

## Quick Start

### 1. CSV File Input
```bash
python binary_classification_evaluator.py \
  --input_type csv \
  --gt_file ground_truth_template.csv \
  --pred_file predictions_template.csv \
  --decision_file decisions_template.csv \
  --output_dir results/
```

### 2. Direct Array Input
```bash
python binary_classification_evaluator.py \
  --input_type arrays \
  --y_true "0,0,1,0,1,1" \
  --y_pred "0.1,0.3,0.8,0.2,0.9,0.7" \
  --y_decision "0,0,1,0,1,1" \
  --output_dir results/
```

## Input Formats

### CSV Files
All CSV files should have the format: `filename,value`

#### Ground Truth (`ground_truth_template.csv`)
```csv
file_001.wav,0
file_002.wav,0
file_003.wav,1
file_004.wav,0
file_005.wav,1
```
- **Column 1**: Filename (any string)
- **Column 2**: True label (0=normal, 1=anomaly)

#### Predictions (`predictions_template.csv`)
```csv
file_001.wav,0.15
file_002.wav,0.23
file_003.wav,0.87
file_004.wav,0.31
file_005.wav,0.92
```
- **Column 1**: Filename (must match ground truth)
- **Column 2**: Continuous anomaly score (0.0 to 1.0+)

#### Decisions (`decisions_template.csv`)
```csv
file_001.wav,0
file_002.wav,0
file_003.wav,1
file_004.wav,0
file_005.wav,1
```
- **Column 1**: Filename (must match ground truth)
- **Column 2**: Binary decision (0=normal, 1=anomaly)

### Array Input
- **`--y_true`**: Comma-separated true labels (0,0,1,0,1,1)
- **`--y_pred`**: Comma-separated continuous scores (0.1,0.3,0.8,0.2,0.9,0.7)
- **`--y_decision`**: Comma-separated binary decisions (0,0,1,0,1,1) [optional]

## Output Formats

### Console Output
```
Binary Classification Metrics:
auc: 1.0000
pauc: 1.0000
pr_auc: 1.0000
precision: 1.0000
recall: 1.0000
f1: 1.0000
accuracy: 1.0000
specificity: 1.0000
true_positive_rate: 1.0000
false_positive_rate: 0.0000
true_negative_rate: 1.0000
false_negative_rate: 0.0000
official_score: 1.0000
auc_jackknife: 1.0000
auc_ci95: 0.0000
pauc_jackknife: 1.0000
pauc_ci95: 0.0000
official_score_ci95: 0.0000
```

### CSV Output (`metrics.csv`)
```csv
auc,pauc,pr_auc,precision,recall,f1,accuracy,specificity,true_positive_rate,false_positive_rate,true_negative_rate,false_negative_rate,official_score,auc_jackknife,auc_ci95,pauc_jackknife,pauc_ci95,official_score_ci95
1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0,0.0,0.0
```

### JSON Output (`metrics.json`)
```json
{
  "auc": 1.0,
  "pauc": 1.0,
  "pr_auc": 1.0,
  "precision": 1.0,
  "recall": 1.0,
  "f1": 1.0,
  "accuracy": 1.0,
  "specificity": 1.0,
  "true_positive_rate": 1.0,
  "false_positive_rate": 0.0,
  "true_negative_rate": 1.0,
  "false_negative_rate": 0.0,
  "official_score": 1.0,
  "auc_jackknife": 1.0,
  "auc_ci95": 0.0,
  "pauc_jackknife": 1.0,
  "pauc_ci95": 0.0,
  "official_score_ci95": 0.0
}
```

## Metrics Explained

### Core DCASE Metrics
- **AUC**: Area Under ROC Curve (0.0 to 1.0)
- **pAUC**: Partial AUC up to max_fpr threshold (default: 0.1)
- **Official Score**: Harmonic mean of AUC and pAUC
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)

### Additional Metrics
- **PR-AUC**: Area Under Precision-Recall Curve
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Specificity**: TN / (TN + FP)
- **True Positive Rate**: Same as Recall
- **False Positive Rate**: FP / (FP + TN)
- **True Negative Rate**: Same as Specificity
- **False Negative Rate**: FN / (FN + TP)

### Statistical Confidence
- **Jackknife CI**: 95% confidence intervals for AUC, pAUC, and official score
- **AUC Jackknife**: Jackknife estimate of AUC
- **pAUC Jackknife**: Jackknife estimate of pAUC

## Command Line Arguments

### Required Arguments
- `--input_type`: Input type (`csv` or `arrays`)

### CSV Input Arguments
- `--gt_file`: Ground truth CSV file
- `--pred_file`: Predictions CSV file
- `--decision_file`: Binary decisions CSV file (optional)

### Array Input Arguments
- `--y_true`: True labels as comma-separated string
- `--y_pred`: Predicted scores as comma-separated string
- `--y_decision`: Binary decisions as comma-separated string (optional)

### Optional Arguments
- `--max_fpr`: Max FPR for pAUC calculation (default: 0.1)
- `--output_dir`: Output directory (default: ./results)

## Example Usage

### Run All Examples
```bash
./example_usage.sh
```

### Individual Examples

#### 1. Basic CSV Evaluation
```bash
python binary_classification_evaluator.py \
  --input_type csv \
  --gt_file ground_truth_template.csv \
  --pred_file predictions_template.csv \
  --decision_file decisions_template.csv \
  --output_dir results/
```

#### 2. Array Input with Custom pAUC
```bash
python binary_classification_evaluator.py \
  --input_type arrays \
  --y_true "0,0,1,0,1,1,0,0,1,0" \
  --y_pred "0.15,0.23,0.87,0.31,0.92,0.78,0.19,0.27,0.85,0.22" \
  --y_decision "0,0,1,0,1,1,0,0,1,0" \
  --max_fpr 0.05 \
  --output_dir results/
```

#### 3. Scores Only (No Binary Decisions)
```bash
python binary_classification_evaluator.py \
  --input_type arrays \
  --y_true "0,0,1,0,1,1,0,0,1,0" \
  --y_pred "0.15,0.23,0.87,0.31,0.92,0.78,0.19,0.27,0.85,0.22" \
  --output_dir results/
```

## DCASE Compatibility

This tool implements the exact evaluation methodology from DCASE Task 2:

- **Harmonic Mean**: Uses `scipy.stats.hmean()` for official score
- **Division Safety**: Uses `numpy.maximum(denominator, sys.float_info.epsilon)`
- **Jackknife Resampling**: Exact implementation from DCASE evaluator
- **pAUC**: Partial AUC with configurable max_fpr (default: 0.1)
- **Confidence Intervals**: 95% CI using t-distribution

## Key Concepts

### Harmonic Mean
More sensitive to low values than arithmetic mean:
```python
# Example: AUC=0.8, pAUC=0.2
arithmetic_mean = (0.8 + 0.2) / 2 = 0.5
harmonic_mean = scipy.stats.hmean([0.8, 0.2]) = 0.32
```

### pAUC (Partial AUC)
AUC calculated only up to a specific False Positive Rate:
- **Regular AUC**: Area under entire ROC curve (FPR: 0 to 1)
- **pAUC**: Area under ROC curve (FPR: 0 to 0.1)

### Jackknife Resampling
Statistical method for confidence intervals:
1. Calculate metric on full dataset
2. For each sample, calculate metric with that sample removed
3. Use leave-one-out estimates to compute standard error
4. Calculate 95% confidence interval

## Files

- `binary_classification_evaluator.py` - Main evaluation script
- `ground_truth_template.csv` - Example ground truth file
- `predictions_template.csv` - Example predictions file
- `decisions_template.csv` - Example decisions file
- `example_usage.sh` - Example usage script
- `README.md` - This documentation

## License

This tool is provided as-is for research and educational purposes.

## Citation

If you use this tool in your research, please cite the DCASE challenge papers:

```
@inproceedings{dcase2025,
  title={Description and discussion on DCASE 2025 challenge task 2: first-shot unsupervised anomalous sound detection for machine condition monitoring},
  author={Nishida, Tomoya and Harada, Noboru and Niizumi, Daisuke and others},
  booktitle={arXiv e-prints: 2506.10097},
  year={2025}
}
```
