# Temperature Scaling for Neural Networks

A paper replication project based on **On Calibration of Modern Neural Networks** by Guo et al. (2017).

This notebook explores a core claim of the paper: a neural network can be **accurate yet poorly calibrated**, and a simple post-processing method, **temperature scaling**, can improve the quality of its predicted probabilities without changing its class ranking.

## Paper
- Guo, Pleiss, Sun, and Weinberger. *On Calibration of Modern Neural Networks*.
- ICML 2017 / PMLR 70.
- Paper link: https://proceedings.mlr.press/v70/guo17a/guo17a.pdf

## Objective
The goal of this project is to reproduce the main calibration workflow from the paper:

1. Train a neural network classifier on **CIFAR-100**.
2. Evaluate its predictive confidence on validation and test sets.
3. Measure calibration using:
   - **Expected Calibration Error (ECE)**
   - **Maximum Calibration Error (MCE)**
   - **Reliability diagrams**
4. Fit a **temperature scaling** parameter on validation logits by minimising **negative log-likelihood (NLL)**.
5. Compare calibration **before** and **after** temperature scaling.

## What has been implemented
From the current notebook, the following pipeline has been completed:

### 1. CIFAR-100 data loading from local pickle files
The dataset is loaded manually from the original CIFAR-100 Python files rather than directly from PyTorch/TensorFlow loaders.

- `train`
- `test`

The code unpickles the raw files and extracts:
- image data: `b'data'`
- labels: `b'fine_labels'`

### 2. Data preprocessing
The raw CIFAR-100 image vectors are reshaped from:

$$
(50000, 3072) \rightarrow (50000, 3, 32, 32)
$$

The images are then normalised to:

$$
[0,1]
$$

A validation split is created from the original training set using `train_test_split`, so calibration is fitted on a validation set rather than on the test set.

### 3. DataLoader preparation
The NumPy arrays are converted into PyTorch tensors and wrapped in `TensorDataset` / `DataLoader` objects for:
- training
- validation
- testing

### 4. Calibration metrics
Two calibration metrics have been implemented manually:

#### Expected Calibration Error (ECE)
$$
\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|
$$

#### Maximum Calibration Error (MCE)
$$
\text{MCE} = \max_m \left| \text{acc}(B_m) - \text{conf}(B_m) \right|
$$

where:
- $B_m$ is the set of predictions in confidence bin $m$
- $\text{acc}(B_m)$ is bin accuracy
- $\text{conf}(B_m)$ is average confidence in the bin

### 5. Reliability diagram
A custom reliability diagram function has been written to visualise the gap between:
- empirical accuracy in each confidence bin
- average model confidence in each confidence bin

This makes the model's overconfidence or underconfidence visible.

### 6. Base model
The current model is:
- **ResNet-50** from `torchvision.models`
- final fully connected layer replaced with a 100-class output layer
- trained from scratch on CIFAR-100

### 7. Training loop
A PyTorch training loop has been implemented using:
- `CrossEntropyLoss`
- SGD with momentum
- weight decay
- multi-step learning rate scheduler

### 8. Logit extraction
A helper function collects logits and labels from:
- validation set
- test set

This is necessary because temperature scaling works by rescaling **logits**, not probabilities.

### 9. Temperature scaling
A custom `TemperatureScaler` class has been implemented.

The workflow is:
1. Take validation logits
2. Divide logits by temperature $T$
3. Apply softmax
4. Compute NLL
5. Optimize $T$ using `scipy.optimize.minimize_scalar`

The calibrated probabilities are computed as:

$$
\text{softmax}(z / T)
$$

where:
- $z$ = logits
- $T > 0$ = learned temperature parameter

### 10. Before vs after comparison
The notebook compares:
- ECE before temperature scaling
- ECE after temperature scaling
- MCE before temperature scaling
- MCE after temperature scaling
- reliability diagrams before and after calibration

## Current learning outcomes
This project already demonstrates several important ideas from the paper:

- **Accuracy and calibration are different concepts**.
- A classifier may be reasonably accurate but still **overconfident**.
- **Temperature scaling** is a lightweight post-processing method.
- Calibration can be evaluated visually and numerically.

## Project structure
Current main artefact:
- `Temperature_Scaling_for_NN.ipynb`

Key components in the notebook:
- CIFAR-100 pickle loader
- preprocessing and validation split
- DataLoader creation
- ECE implementation
- MCE implementation
- reliability diagram plotting
- ResNet-50 training
- logit extraction
- temperature scaling via validation NLL
- before/after calibration evaluation

## Dependencies
Based on the notebook, the current project uses:
- `numpy`
- `pandas`
- `matplotlib`
- `torch`
- `torchvision`
- `scikit-learn`
- `scipy`

## Notes
- The model is currently trained **from scratch** rather than loaded from pretrained weights.
- ResNet-50 on CIFAR-100 is computationally heavier than smaller baselines.
- This is currently focused on reproducing the **calibration pipeline**, rather than fully reproducing every experiment in the paper.


## Summary
This project is a hands-on replication of the temperature scaling idea from Guo et al. (2017). It loads CIFAR-100 manually, trains a ResNet classifier, computes calibration metrics, fits a temperature parameter on validation logits, and evaluates how calibration improves after scaling.

In short: the notebook is not just training a classifier; it is studying whether the classifier's **confidence scores are trustworthy**.
