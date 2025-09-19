#!/usr/bin/env python3
"""
General Binary Classification Evaluator
Supports both CSV files and direct array input
"""

import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
import scipy.stats
import json

def load_csv_data(file_path, score_col=1, filename_col=0):
    """Load CSV with filename,score format"""
    df = pd.read_csv(file_path, header=None)
    return df[score_col].values

def load_arrays(y_true_str, y_pred_str, y_decision_str=None):
    """Load arrays from comma-separated strings"""
    y_true = np.array([float(x) for x in y_true_str.split(',')])
    y_pred = np.array([float(x) for x in y_pred_str.split(',')])
    y_decision = None
    if y_decision_str:
        y_decision = np.array([int(x) for x in y_decision_str.split(',')])
    return y_true, y_pred, y_decision

def jackknife_estimate(fn, var_list):
    """Jackknife resampling for confidence intervals (from DCASE evaluator)"""
    def removed_i(var_list, remove_i):
        return [v[[i for i in range(len(v)) if i != remove_i]] for v in var_list]
    var_list = [np.array(v) for v in var_list]
    N = len(var_list[0])
    # (1)
    theta_hat = fn(*var_list)
    # (2)
    thetai_hats = [fn(*removed_i(var_list, i)) for i in range(N)]
    # (3)
    theta_hat_mean = np.mean(thetai_hats)
    # (4)
    thetai_tildes = [N * theta_hat - (N - 1) * thetai_hat for thetai_hat in thetai_hats]
    # (5)
    theta_hat_jack = np.mean(thetai_tildes)
    # (6)
    sigma_hat_jack = np.sqrt(np.sum([(thi - theta_hat_mean)**2 for thi in thetai_hats]) / (N * (N-1)))
    # (7) - CI only
    confidence = 0.95
    dof = N - 1
    t_crit = np.abs(scipy.stats.t.ppf((1 - confidence) / 2, dof))
    ci95_jack = t_crit * sigma_hat_jack

    return theta_hat_jack, ci95_jack

def calculate_metrics(y_true, y_pred, y_decision=None, max_fpr=0.1):
    """Calculate all binary classification metrics (DCASE-compatible)"""
    import sys
    metrics_dict = {}
    
    # AUC metrics (exactly like DCASE)
    metrics_dict['auc'] = metrics.roc_auc_score(y_true, y_pred)
    metrics_dict['pauc'] = metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
    
    # Precision-Recall AUC (additional metric)
    metrics_dict['pr_auc'] = metrics.average_precision_score(y_true, y_pred)
    
    # Binary classification metrics (if decisions provided) - EXACTLY like DCASE
    if y_decision is not None:
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_decision).ravel()
        # Use numpy.maximum with sys.float_info.epsilon like DCASE
        prec = tp / np.maximum(tp + fp, sys.float_info.epsilon)
        recall = tp / np.maximum(tp + fn, sys.float_info.epsilon)
        f1 = 2.0 * prec * recall / np.maximum(prec + recall, sys.float_info.epsilon)
        
        metrics_dict['precision'] = prec
        metrics_dict['recall'] = recall
        metrics_dict['f1'] = f1
        metrics_dict['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
        metrics_dict['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Additional DCASE-style metrics
        metrics_dict['true_positive_rate'] = recall  # Same as recall
        metrics_dict['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics_dict['true_negative_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics_dict['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Official score calculation (like DCASE) - harmonic mean of AUC and pAUC
    all_perf = np.array([metrics_dict['auc'], metrics_dict['pauc']], dtype=float)
    metrics_dict['official_score'] = scipy.stats.hmean(np.maximum(all_perf, sys.float_info.epsilon), axis=None)
    
    # Jackknife confidence intervals (like DCASE)
    try:
        auc_jack, auc_ci95 = jackknife_estimate(fn=metrics.roc_auc_score, var_list=[y_true, y_pred])
        pauc_jack, pauc_ci95 = jackknife_estimate(fn=lambda a,b: metrics.roc_auc_score(a, b, max_fpr=max_fpr), var_list=[y_true, y_pred])
        metrics_dict['auc_jackknife'] = auc_jack
        metrics_dict['auc_ci95'] = auc_ci95
        metrics_dict['pauc_jackknife'] = pauc_jack
        metrics_dict['pauc_ci95'] = pauc_ci95
        metrics_dict['official_score_ci95'] = np.mean([auc_ci95, pauc_ci95])
    except Exception as e:
        print(f"Warning: Could not calculate jackknife CI: {e}")
    
    return metrics_dict

def main():
    parser = argparse.ArgumentParser(description='Binary Classification Evaluator')
    parser.add_argument('--input_type', choices=['csv', 'arrays'], required=True,
                        help='Input type: csv files or direct arrays')
    
    # CSV input
    parser.add_argument('--gt_file', help='Ground truth CSV file')
    parser.add_argument('--pred_file', help='Predictions CSV file')
    parser.add_argument('--decision_file', help='Binary decisions CSV file (optional)')
    
    # Array input
    parser.add_argument('--y_true', help='True labels as comma-separated string')
    parser.add_argument('--y_pred', help='Predicted scores as comma-separated string')
    parser.add_argument('--y_decision', help='Binary decisions as comma-separated string (optional)')
    
    # Parameters
    parser.add_argument('--max_fpr', type=float, default=0.1,
                        help='Max FPR for pAUC calculation')
    parser.add_argument('--output_dir', default='./results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Load data
    if args.input_type == 'csv':
        y_true = load_csv_data(args.gt_file)
        y_pred = load_csv_data(args.pred_file)
        y_decision = load_csv_data(args.decision_file) if args.decision_file else None
    else:
        y_true, y_pred, y_decision = load_arrays(args.y_true, args.y_pred, args.y_decision)
    
    # Calculate metrics
    metrics_dict = calculate_metrics(y_true, y_pred, y_decision, args.max_fpr)
    
    # Save results
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save as JSON
    with open(f'{args.output_dir}/metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    # Save as CSV
    pd.DataFrame([metrics_dict]).to_csv(f'{args.output_dir}/metrics.csv', index=False)
    
    # Print results
    print("Binary Classification Metrics:")
    for metric, value in metrics_dict.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()