#!/bin/bash

# Binary Classification Evaluator - Example Usage Commands

echo "=== Example 1: CSV File Input ==="
python binary_classification_evaluator.py \
  --input_type csv \
  --gt_file ground_truth_template.csv \
  --pred_file predictions_template.csv \
  --decision_file decisions_template.csv \
  --output_dir results_csv

echo ""
echo "=== Example 2: Direct Array Input ==="
python binary_classification_evaluator.py \
  --input_type arrays \
  --y_true "0,0,1,0,1,1,0,0,1,0" \
  --y_pred "0.15,0.23,0.87,0.31,0.92,0.78,0.19,0.27,0.85,0.22" \
  --y_decision "0,0,1,0,1,1,0,0,1,0" \
  --output_dir results_arrays

echo ""
echo "=== Example 3: Only Continuous Scores (no binary decisions) ==="
python binary_classification_evaluator.py \
  --input_type arrays \
  --y_true "0,0,1,0,1,1,0,0,1,0" \
  --y_pred "0.15,0.23,0.87,0.31,0.92,0.78,0.19,0.27,0.85,0.22" \
  --output_dir results_scores_only

echo ""
echo "=== Example 4: Custom pAUC threshold ==="
python binary_classification_evaluator.py \
  --input_type arrays \
  --y_true "0,0,1,0,1,1,0,0,1,0" \
  --y_pred "0.15,0.23,0.87,0.31,0.92,0.78,0.19,0.27,0.85,0.22" \
  --y_decision "0,0,1,0,1,1,0,0,1,0" \
  --max_fpr 0.05 \
  --output_dir results_custom_pauc

echo ""
echo "=== Example 5: DCASE-style evaluation ==="
python binary_classification_evaluator.py \
  --input_type csv \
  --gt_file ground_truth_template.csv \
  --pred_file predictions_template.csv \
  --decision_file decisions_template.csv \
  --max_fpr 0.1 \
  --output_dir results_dcase_style

echo ""
echo "All examples completed! Check the output directories for results."
