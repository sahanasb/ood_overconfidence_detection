# Data Preprocessing for OOD Overconfidence Detection

## Overview

This notebook prepares the **NSL-KDD (ID)** and **UNSW-NB15 (OOD)** datasets for the OOD overconfidence detection project.

The preprocessing pipeline includes:

- Data loading and cleaning
- Exploratory Data Analysis (EDA)
- Feature alignment
- Train / Validation / Test split
- Feature scaling
- One-Hot encoding


## Dataset Download

### 1. NSL-KDD (In-Distribution)

The NSL-KDD dataset is downloaded automatically by the notebook, so no manual download is required.

Files used:

- `KDDTrain+.ARFF`
- `KDDTest+.ARFF`

---

Please download the following datasets before running the notebook.


### 2. UNSW-NB15 (Out-of-Distribution)

Download:

(https://www.kaggle.com/datasets/primus11/unsw-nb15-dataset)

Files used:

- `unsw_test.csv`
- `unsw_train.csv`

---

## Datasets

### In-Distribution (ID)

- **Dataset:** NSL-KDD
- **Training:** KDDTrain+
- **Testing:** KDDTest+

The official NSL-KDD training set is further split into:

- Training
- Validation (used for Temperature Scaling)


---

### Out-of-Distribution (OOD)

- **Dataset:** UNSW-NB15

The official train/test split is preserved.

---

## Feature Alignment

The following semantically aligned features are used:

| NSL-KDD | UNSW-NB15 |
|----------|-----------|
| duration | dur |
| src_bytes | sbytes |
| dst_bytes | dbytes |
| protocol_type | proto |
| service | service |
| dst_host_count | ct_dst_ltm |
| dst_host_srv_count | ct_srv_dst |

### Excluded Feature

`flag` (NSL-KDD) and `state` (UNSW-NB15) were **not used** because their definitions and encoding schemes are different, making reliable feature alignment impossible.

---


## Preprocessing Pipeline

### Numerical Features

- StandardScaler
- **Fit only on the NSL-KDD training set**
- Reused for:
  - Validation
  - Test
  - OOD Train
  - OOD Test

### Categorical Features

- OneHotEncoder
- `handle_unknown="ignore"`
- Fit only on the NSL-KDD training set

Unknown protocol/service categories in UNSW-NB15 are encoded without creating new feature dimensions.

This guarantees that **all datasets share the same feature space**.

---

## Output Variables

### DataFrames (before preprocessing)

Useful for visualization or future preprocessing.

```python
X_train_df
X_val_df
X_test_df

X_ood_train_df
X_ood_test_df
```

### Model Inputs (after preprocessing)

These arrays should be used for model training.

```python
X_train
X_val
X_test

X_ood_train
X_ood_test
```

### Labels

```python
y_train
y_val
y_test

y_ood_train
y_ood_test
```

---

### Layer 1 Classifier

```text
X_train + y_train
        ↓
Train classifier
```

### Temperature Scaling

```text
X_val + y_val
      ↓
Learn temperature T
```

### Final Evaluation

```text
X_test + y_test
```

### OOD Evaluation

```text
X_ood_train
X_ood_test
```

Generate:

- Confidence
- Entropy
- Margin

These fingerprint features will be used as inputs to Layer 2.

---

## Notes

- The preprocessing pipeline is fitted **only on the NSL-KDD training set** to avoid data leakage.
- Feature alignment creates a common feature space between NSL-KDD and UNSW-NB15.
- Unknown categorical values are handled by `OneHotEncoder(handle_unknown="ignore")`.
- All processed datasets have identical feature dimensions.

---
