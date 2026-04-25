Privacy-Enhancing Techniques for AI Models  
Federated Learning & Differential Privacy

Team Members
- Abdul Gafoor  
- Ahmar Shafith  
- Harshith Mahendra  

Project Overview

This project explores **privacy-preserving machine learning techniques** by comparing three training approaches:

1. Centralized Training (Baseline CNN)
2. Federated Learning (FL)
3. Federated Learning with Differential Privacy (FL + DP)

Using the MNIST dataset, we analyze the **tradeoff between accuracy, privacy, and performance**.


Problem Statement

Traditional machine learning requires **centralizing sensitive data**, which introduces major risks:

- Data breaches and single point of failure
- Privacy leakage from trained models
- Re-identification of anonymized data
- Regulatory non-compliance (GDPR, HIPAA)

This project investigates how to train models **without exposing raw user data**.

Approaches Implemented

1. Baseline Model (Centralized Learning)
- CNN trained on full dataset
- Standard supervised learning
- Establishes maximum achievable accuracy

2. Federated Learning (FL)
- Data distributed across multiple clients
- Each client trains locally
- Only model updates are shared (not raw data)
- Aggregation using **FedAvg algorithm**

*Privacy Benefit:* No centralized data storage

3. Federated Learning + Differential Privacy (FL + DP)
- Adds **formal privacy guarantees**
- Uses Gaussian noise on gradients
- Implemented using **Opacus**

*Privacy Benefit:* Prevents leakage of individual training data

Model Architecture

- Input: 28×28 grayscale images (MNIST)
- 2 Convolutional layers (16 & 32 filters)
- ReLU + MaxPooling
- Fully connected layer (128 units)
- Output: 10 classes (digits)

Results Summary

| Model | Accuracy | Training Time | Privacy Level |
|------|---------|--------------|--------------|
| Baseline CNN | 99.07% | 180s |  None |
| Federated Learning | 98.87% | 225s |  Partial |
| FL + Differential Privacy | 88.62% | 571s |  Strong |

Key Insights

-  Federated Learning retains **~99% accuracy** without sharing data  
-  Differential Privacy provides **mathematical privacy guarantees**  
-  Strong privacy comes with **accuracy and performance cost**  

Privacy & Security Concepts

- Data minimization (no raw data sharing)
- Gradient sharing instead of data sharing
- Differential Privacy (ε = 1.0, δ = 1e-5)
- Gradient clipping + noise injection

Trade-Off Analysis

| Factor | Baseline | FL | FL + DP |
|--------|---------|----|--------|
| Accuracy | High | Very High | Moderate |
| Privacy | None | Medium | Strong |
| Performance | Fast | Medium | Slow |



## 📁 Project Structure
