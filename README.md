
# Customer Segmentation Using Unsupervised Learning

## Overview

This project explores unsupervised machine learning techniques
using a synthetic customer dataset.

The objective is to discover customer patterns without using
predefined labels.

## Technologies

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Dataset

The project generates 600 synthetic customer records.

Features:

- monthly_spending
- transactions_per_month
- account_age_months

The dataset is generated without labels.

## Machine Learning Experiments

### 1. K-Means Clustering

Groups customers into clusters based on feature similarity.

### 2. Cluster Count Comparison

Compares different K-Means cluster counts using:

- Inertia
- Silhouette score
- Elbow method

### 3. DBSCAN

Uses density-based clustering to identify customer groups
and noise points.

### 4. Anomaly Detection

Uses Isolation Forest to identify unusual customer records.

### 5. Principal Component Analysis

Reduces three original features into two principal components
for visualization.

## Installation

Create a virtual environment:

```bash
python -m venv venv
```
