# Import Necessary Libraries
import pandas as pd
import torch
import torch.nn as nn
from torchsummary import summary
from custom_hyperspectral_dataset import Hspectral_Dtatset

# Create Feed Forward Network Class
"""
// Very Basic CNN Model version 0.1
"""
# Define the CNN model
class CustomCNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(16 * 101, 64)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        return x

# Let's Run and Create Model's Instance
if __name__ == "__main__":
    # Define the dataset path
    CSV_FILE = "../dataset/hyperspectral.csv"
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create an instance of the dataset
    dataset = Hspectral_Dtatset(CSV_FILE)
    print(dataset.data.shape[1])

    # Get the input dimension from the dataset
    input_dim = dataset.data.shape[1]
    # Define the output dimension
    output_dim = 2

    # Create an instance of the CNN model
    model = CustomCNN(input_dim, output_dim).to(device)

    # Print the model summary
    summary(model, input_size=(input_dim,))

    # Move the model to the device
    model.to(device)



"""
// For checking and printing each and every lines
"""
"""    
    input_dim = 4
    output_dim = 2
    model = CustomCNN_V1(input_dim, output_dim)

    # Create a sample input tensor
    input_tensor = torch.randn(2, input_dim)
    print("Input shape:", input_tensor.shape)

    # Perform the forward pass
    x = input_tensor
    x = x.unsqueeze(1)
    print("After unsqueeze(1) - Input shape:", x.shape)

    x = model.conv1(x)
    print("After conv1 - Output shape:", x.shape)

    x = model.relu1(x)
    print("After relu1 - Output shape:", x.shape)

    x = model.pool1(x)
    print("After pool1 - Output shape:", x.shape)

    x = x.view(x.size(0), -1)
    print("After view - Output shape:", x.shape)

    x = model.fc1(x)
    print("After fc1 - Output shape:", x.shape)

    x = model.relu2(x)
    print("After relu2 - Output shape:", x.shape)

    x = model.fc2(x)
    print("After fc2 - Output shape:", x.shape)
    """


