import logging  # Importing the logging module for configurable message logging.
from matplotlib import pyplot as plt  # Importing matplotlib for plotting.
import numpy as np  # Importing numpy for numerical operations.
import seaborn as sns  # Importing seaborn for advanced data visualization.
from sklearn.metrics import confusion_matrix  # Importing confusion matrix computation from sklearn.
import torch  # Importing PyTorch for machine learning and deep learning operations.
import torch.nn as nn  # Importing the neural network module from PyTorch.
import torch.optim as optim  # Importing the optimization module from PyTorch.
from torch.utils.data import DataLoader  # Importing DataLoader to handle batches of data.

# Dictionary to configure backend setup (not actively used in this code).
backend_setup = {"init_args": {"logging_level": logging.ERROR, "log_to_driver": False}}

# Custom logging filter to allow only INFO level logs.
class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.INFO  # Only log INFO level messages.

# Function to configure the training based on the server round.
def fit_config(server_round: int):
    config_dict = {
        "local_epochs": 2 if server_round < 3 else 5,  # Use fewer epochs for early rounds.
    }
    return config_dict

# Defining a simple feedforward neural network model.
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()  # Initializing the parent class (nn.Module).
        self.fc = nn.Linear(784, 128)  # First fully connected layer (input: 784, output: 128).
        self.relu = nn.ReLU()  # Activation function (ReLU).
        self.out = nn.Linear(128, 10)  # Output layer (input: 128, output: 10 classes).

    # Defining the forward pass of the model.
    def forward(self, x):
        x = torch.flatten(x, 1)  # Flattening input tensor to 1D (except batch dimension).
        x = self.fc(x)  # Passing input through the first fully connected layer.
        x = self.relu(x)  # Applying ReLU activation.
        x = self.out(x)  # Passing data through the output layer.
        return x

# Function to train the model using a given training dataset.
def train_model(model, train_set):
    batch_size = 64  # Number of samples per training batch.
    num_epochs = 10  # Number of training epochs.

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)  # DataLoader for batching and shuffling.

    criterion = nn.CrossEntropyLoss()  # Loss function for classification tasks.
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # Stochastic Gradient Descent optimizer.

    model.train()  # Setting the model to training mode.
    for epoch in range(num_epochs):  # Looping through epochs.
        running_loss = 0.0  # Accumulator for tracking loss.
        for inputs, labels in train_loader:  # Looping through data batches.
            optimizer.zero_grad()  # Clearing gradients from the previous step.
            outputs = model(inputs)  # Forward pass through the model.
            loss = criterion(outputs, labels)  # Calculating loss.
            loss.backward()  # Backpropagation.
            optimizer.step()  # Updating model weights.
            running_loss += loss.item()  # Adding batch loss to running loss.

# Function to evaluate the model using a test dataset.
def evaluate_model(model, test_set):
    model.eval()  # Setting the model to evaluation mode.
    correct = 0  # Counter for correctly predicted samples.
    total = 0  # Total number of samples.
    total_loss = 0  # Accumulator for tracking total loss.

    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)  # DataLoader for batching (no shuffling).
    criterion = nn.CrossEntropyLoss()  # Loss function for evaluation.

    with torch.no_grad():  # No gradient computation during evaluation.
        for inputs, labels in test_loader:  # Looping through data batches.
            outputs = model(inputs)  # Forward pass through the model.
            _, predicted = torch.max(outputs.data, 1)  # Getting class with the highest probability.
            total += labels.size(0)  # Incrementing total samples.
            correct += (predicted == labels).sum().item()  # Counting correctly predicted samples.

            loss = criterion(outputs, labels)  # Calculating loss for the batch.
            total_loss += loss.item()  # Adding batch loss to total loss.

    accuracy = correct / total  # Calculating accuracy.
    average_loss = total_loss / len(test_loader)  # Calculating average loss.
    return average_loss, accuracy  # Returning loss and accuracy.

# Function to compute the confusion matrix for the model's predictions.
def compute_confusion_matrix(model, testset):
    true_labels = []  # List to store true labels.
    predicted_labels = []  # List to store predicted labels.

    # Iterating through test dataset to generate predictions.
    for image, label in testset:
        output = model(image.unsqueeze(0))  # Forward pass with batch dimension added.
        _, predicted = torch.max(output, 1)  # Getting class with the highest probability.

        true_labels.append(label)  # Appending true label.
        predicted_labels.append(predicted.item())  # Appending predicted label.

    true_labels = np.array(true_labels)  # Converting true labels to numpy array.
    predicted_labels = np.array(predicted_labels)  # Converting predicted labels to numpy array.

    cm = confusion_matrix(true_labels, predicted_labels)  # Computing confusion matrix.

    return cm  # Returning confusion matrix.

# Function to plot the confusion matrix using seaborn.
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 4))  # Setting figure size.
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", linewidths=0.5)  # Plotting heatmap.
    plt.title(title)  # Setting title for the plot.
    plt.xlabel("Predicted Label")  # Label for x-axis.
    plt.ylabel("True Label")  # Label for y-axis.
    plt.show()  # Displaying the plot.
