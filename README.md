# Credit Card Fraud Detection with Neural Networks

This project explores the use of deep learning for detecting fraudulent credit card transactions under highly imbalanced conditions.

## Motivation

The original iteration of this project used the *Credit Card Fraud Detection Dataset 2023*, which contains a relatively more balanced distribution of fraud cases due to prior preprocessing and sampling strategies.

While this setup was useful for initial experimentation, it did not accurately reflect real-world fraud detection scenarios, where fraudulent transactions are extremely rare.

To address this limitation, I transitioned to the classic **Credit Card Fraud Detection dataset (MLG-UCB)**, which is significantly more imbalanced (~0.17% fraud rate). This shift was intentional in order to evaluate whether the model could generalize under realistic and more challenging conditions.

## Dataset

- Source: Kaggle – Credit Card Fraud Detection (MLG-UCB)
- Features: PCA-transformed anonymized variables (V1–V28), Amount
- Target: Binary classification (Fraud / Non-Fraud)
- Class distribution: Highly imbalanced (~0.17% fraud cases)

## Approach

A neural network built with TensorFlow/Keras was used as the base model, incorporating:

- Standard scaling for numerical stability
- Class weighting to address imbalance
- Dropout layers for regularization
- Threshold tuning using precision-recall analysis

## Evaluation Strategy

Given the extreme class imbalance, traditional accuracy is not a meaningful metric. The model is primarily evaluated using:

- Precision-Recall AUC (PR-AUC)
- Recall (fraud detection rate)
- Precision (false positive control)
- Threshold-dependent classification metrics

## Results

- PR-AUC: ~0.69
- Fraud Recall: ~0.90
- Fraud Precision: ~0.11 (threshold-dependent)

The model demonstrates strong recall performance, successfully identifying most fraudulent transactions, while precision remains limited due to the aggressive detection strategy required in highly imbalanced settings.

## Key Takeaways

- Class imbalance has a significant impact on model behavior and must be explicitly handled.
- High recall can be achieved at the cost of precision depending on decision thresholds.
- PR-AUC is a more informative metric than accuracy in this context.
- Neural networks are viable, but not necessarily optimal for tabular fraud detection compared to gradient boosting methods.

## Future Work

- Comparison with tree-based models (XGBoost / LightGBM)
- Probability calibration techniques (Platt scaling / isotonic regression)
- Cost-sensitive optimization based on business impact
- Feature engineering beyond PCA-transformed inputs
