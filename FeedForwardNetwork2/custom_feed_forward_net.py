# Import Necessary Libraries
import torch
import torch.nn as nn
from torch.nn import Dropout, BatchNorm1d
import pandas as pd
from torchsummary import summary

# Create Feed Forward Network Class
"""
// Very Basic FFN Model version 0.1
"""
class CustomFFNv1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CustomFFNv1, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


"""
// FFN Model version 0.2

Changes:
1. Add more 3 Fully connected layer total 5
2. Add Linear Layer
"""
class CustomFFNv2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CustomFFNv2, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc5 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        return x

"""
// FFN Model version 0.3

Changes:
1. Fully connected layer 5
2. Add Linear Layer
3. Applied regularization techniques
    a) L1 or L2 Regularization: Import the 'l1_loss' or 'l2_loss' function from torch.nn.functional
    b) Add the regularization term to the loss function during training
    Example:
        l2_lambda = 0.001  # Regularization coefficient
        model = CustomFFNv3(input_dim, hidden_dim, output_dim, l2_lambda)
        loss = loss_function(outputs, labels) + model.l2_regularization_loss()
    c) Adjust the regularization strength by multiplying it with a suitable regularization coefficient
    d) Here I applied L2
    e) Dropout:
        - Import the Dropout module from torch.nn
        - Apply dropout layers after specific linear layers
        - During training, enable dropout using the model.train() method
        - During evaluation, disable dropout using the model.eval() method
        - We can change Dropout values 0.2,0.3,0.5
    f) Batch Normalization:
        - Import the `BatchNorm1d` module from `torch.nn
        - Apply batch normalization layers after specific linear layers.
        - During training, enable batch normalization using the `model.train()` method.
        - During evaluation, disable dropout using the model.eval() method
"""

class CustomFFNv3(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, l2_lambda):
        super(CustomFFNv3, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # Apply Batch Normalization
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)  # Apply Batch Normalization
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)  # Apply Batch Normalization
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)  # Apply Batch Normalization
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.2)

        self.fc5 = nn.Linear(hidden_dim, output_dim)
        self.l2_lambda = l2_lambda

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)  # Apply Batch Normalization
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)  # Apply Batch Normalization
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)  # Apply Batch Normalization
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        x = self.bn4(x)  # Apply Batch Normalization
        x = self.relu4(x)
        x = self.dropout4(x)

        x = self.fc5(x)
        return x


    def l2_regularization_loss(self):
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.sum(torch.square(param))

        return self.l2_lambda * l2_loss

"""
// FFN Model version 0.4

Main Changes:
1. Specify different neurons in different layers increasing 2 times

"""

class CustomFFNv4(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, l2_lambda):
        super(CustomFFNv4, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 32
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # Apply Batch Normalization
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(hidden_dim, 2 * hidden_dim)  # (32, 64)
        self.bn2 = nn.BatchNorm1d(2 * hidden_dim)  # Apply Batch Normalization
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(2 * hidden_dim, 4 * hidden_dim)  # (64, 128)
        self.bn3 = nn.BatchNorm1d(4 * hidden_dim)  # Apply Batch Normalization
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(4 * hidden_dim, 8 * hidden_dim)  # (128, 256)
        self.bn4 = nn.BatchNorm1d(8 * hidden_dim)  # Apply Batch Normalization
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.2)

        self.fc5 = nn.Linear(8 * hidden_dim, output_dim)  # (256, 2=Two Class)
        self.l2_lambda = l2_lambda

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)  # Apply Batch Normalization
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)  # Apply Batch Normalization
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)  # Apply Batch Normalization
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        x = self.bn4(x)  # Apply Batch Normalization
        x = self.relu4(x)
        x = self.dropout4(x)

        x = self.fc5(x)
        return x


    def l2_regularization_loss(self):
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.sum(torch.square(param))

        return self.l2_lambda * l2_loss

"""
// FFN Model version 0.5

Main Changes:
1. Increasing Dimensionality

"""

class CustomFFNv5(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, l2_lambda):
        super(CustomFFNv5, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim) # 32
        self.bn1 = nn.BatchNorm1d(hidden_dim) # Apply Batch Normalization
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(hidden_dim, 2 * hidden_dim) # (32, 64)
        self.bn2 = nn.BatchNorm1d(2 * hidden_dim) # Apply Batch Normalization # 64
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(2 * hidden_dim, 4 * hidden_dim) # (64, 128)
        self.bn3 = nn.BatchNorm1d(4 * hidden_dim) # Apply Batch Normalization
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(4 * hidden_dim, 8 * hidden_dim) # (128, 256)
        self.bn4 = nn.BatchNorm1d(8 * hidden_dim) # Apply Batch Normalization
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.2)

        self.fc5 = nn.Linear(8 * hidden_dim, 16 * hidden_dim) # (256, 512)
        self.bn5 = nn.BatchNorm1d(16 * hidden_dim) # Apply Batch Normalization
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(0.2)

        self.fc6 = nn.Linear(16 * hidden_dim, output_dim)  # (512, 2=Two Class)
        self.l2_lambda = l2_lambda

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)  # Apply Batch Normalization
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)  # Apply Batch Normalization
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)  # Apply Batch Normalization
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        x = self.bn4(x)  # Apply Batch Normalization
        x = self.relu4(x)
        x = self.dropout4(x)

        x = self.fc5(x)
        x = self.bn5(x)  # Apply Batch Normalization
        x = self.relu5(x)
        x = self.dropout5(x)

        x = self.fc6(x)
        return x

    def l2_regularization_loss(self):
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.sum(torch.square(param))

        return self.l2_lambda * l2_loss

"""
// FFN Model version 0.6

Main Changes:
1. Dimentionality Reduction

"""

class CustomFFNv6(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, l2_lambda):
        super(CustomFFNv6, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # Apply Batch Normalization
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)  # Apply Batch Normalization
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)  # Apply Batch Normalization
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(hidden_dim // 4, hidden_dim // 8)
        self.bn4 = nn.BatchNorm1d(hidden_dim // 8)  # Apply Batch Normalization
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.2)

        self.fc5 = nn.Linear(hidden_dim // 8, output_dim)
        self.l2_lambda = l2_lambda

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)  # Apply Batch Normalization
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)  # Apply Batch Normalization
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)  # Apply Batch Normalization
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        x = self.bn4(x)  # Apply Batch Normalization
        x = self.relu4(x)
        x = self.dropout4(x)

        x = self.fc5(x)
        return x

    def l2_regularization_loss(self):
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.sum(torch.square(param))

        return self.l2_lambda * l2_loss

"""
// FFN Model version 0.7

Main Changes:
1. Dimensionality Reduction with more layer

"""

class CustomFFNv7(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, l2_lambda):
        super(CustomFFNv7, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 512
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # Apply Batch Normalization

        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)  # (512, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)  # Apply Batch Normalization

        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)  # (256, 128)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)  # Apply Batch Normalization

        self.fc4 = nn.Linear(hidden_dim // 4, hidden_dim // 8)  # (128, 64)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.2)
        self.bn4 = nn.BatchNorm1d(hidden_dim // 8)  # Apply Batch Normalization

        self.fc5 = nn.Linear(hidden_dim // 8, hidden_dim // 16)  # (64, 32)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(0.2)
        self.bn5 = nn.BatchNorm1d(hidden_dim // 16)  # Apply Batch Normalization

        self.fc6 = nn.Linear(hidden_dim // 16, output_dim)  # (32, 2=Two Class)
        self.l2_lambda = l2_lambda

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.bn1(x)  # Apply Batch Normalization
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.bn2(x)  # Apply Batch Normalization
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.relu3(x)
        x = self.bn3(x)  # Apply Batch Normalization
        x = self.dropout3(x)

        x = self.fc4(x)
        x = self.relu4(x)
        x = self.bn4(x)  # Apply Batch Normalization
        x = self.dropout4(x)

        x = self.fc5(x)
        x = self.relu5(x)
        x = self.bn5(x)  # Apply Batch Normalization
        x = self.dropout5(x)

        x = self.fc6(x)
        return x

    def l2_regularization_loss(self):
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.sum(torch.square(param))

        return self.l2_lambda * l2_loss

# Let's Run and Create Model's Instance
if __name__ == "__main__":
    # Device agnostic
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    # Read CVS file path and read input dimension
    csv = pd.read_csv("../dataset/hyperspectral.csv")
    input_dim = int(len(csv.iloc[1, 1:-1]))
    hidden_dim = 32
    output_dim =2
    l2_lambda = 0.001  # Regularization coefficient

    # Create Model's Instance
    model = CustomFFNv5(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        l2_lambda=l2_lambda
    ).to(device)

    print(f"=================")
    print(f"Model Version 0.5:")
    print(f"=================")

    # Let's print model summary
    summary(model, (input_dim,))