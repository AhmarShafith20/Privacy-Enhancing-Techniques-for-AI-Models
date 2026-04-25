Privacy-Enhancing AI
Federated Learning & Differential Privacy

Overview
This project compares three training approaches on the MNIST dataset to study the accuracy vs privacy tradeoff:
Centralized CNN (Baseline)
Federated Learning (FL)
Federated Learning + Differential Privacy (FL + DP)

Approach
1. Baseline CNN
Trained on full dataset
5 epochs, Adam optimizer
Serves as accuracy benchmark

2. Federated Learning
5 clients, data stays local
Uses FedAvg algorithm
No raw data sharing

3. FL + Differential Privacy
Gaussian noise added to gradients
ε = 1.0, δ = 1e-5
Provides strong privacy guarantees

Model
CNN with 2 conv layers + 1 FC layer
Input: 28×28 MNIST images
Output: 10 classes

Results
Model	Accuracy	Time	Privacy
Baseline	99.07%	180s	None
FL	98.87%	225s	Partial
FL + DP	88.62%	571s	Strong

Key Takeaways
FL keeps accuracy almost equal to baseline
DP adds strong privacy but reduces accuracy
Higher privacy → more noise → lower performance

Team
Abdul Gafoor
Ahmar Shafith
Harshith Mahendra
