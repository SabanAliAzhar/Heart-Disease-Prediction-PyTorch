# Heart-Disease-Prediction-PyTorch
A binary classification model using PyTorch to predict heart disease from clinical data. Implements a neural network with real-time training loss visualization and confusion matrix evaluation.

# Heart Disease Prediction using PyTorch

This repository contains a binary classification project that predicts the presence of heart disease based on patient clinical data. The model is implemented using **PyTorch**, and includes features such as data preprocessing, scaling, training loss visualization, and performance evaluation using confusion matrix and accuracy.

---

## Features

- Preprocessing of the heart disease dataset
- Standardization of features using `StandardScaler`
- Conversion of dataset to PyTorch tensors
- Neural network model with one hidden layer using ReLU activation
- Binary classification using Sigmoid activation and BCELoss
- Training with Adam optimizer
- Real-time tracking of training and test loss
- Visualization of training loss over epochs
- Confusion matrix and test accuracy evaluation

---

## Dataset

The dataset should be in **Excel format** with the following columns:

- `num`: Target value (presence of heart disease)
- Clinical features (e.g., age, sex, cholesterol, etc.)
- `Unnamed: 0`: Optional index column (will be dropped)

**Target Creation:**  
`target = 1` if `num > 0` else `0` (binary classification)

---

## Installation

1. Clone this repository:

```bash
[git clone https://github.com/SabanAliAzhar/Heart-Disease-Prediction-PyTorch]

Install required packages:

bash
Copy code
pip install pandas scikit-learn torch matplotlib seaborn 
Usage
Load your dataset:

python
Copy code
import pandas as pd
df = pd.read_excel("heart_dataset.xlsx")

The script will:

Preprocess data

Train the neural network

Plot training loss

Display a confusion matrix

Print test accuracy

Neural Network Architecture
yaml
Copy code
Input Layer: 13 features
Hidden Layer: 16 neurons, ReLU activation
Output Layer: 1 neuron, Sigmoid activation
Loss Function: Binary Cross Entropy Loss
Optimizer: Adam
Results
Training loss is plotted over epochs to track learning

Confusion matrix visualizes true/false positives/negatives

Test accuracy provides the model performance on unseen data
