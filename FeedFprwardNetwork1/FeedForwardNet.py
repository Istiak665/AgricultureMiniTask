import torch
import torch.nn as nn
import pandas as pd
from torchsummary import summary

# Define the FNN
class FNNv1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FNNv1, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Adding Additional Fully Connected Layer
class FNNv2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FNNv2, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

'''
Modified network that introduces more complexity by increasing the hidden layer
dimensions and adding dropout regularization
'''

class FNNv3(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FNNv3, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)  # Dropout regularization with 50% probability
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)  # Dropout regularization with 50% probability
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

'''
In this modified architecture (FNNv2), we have added two additional hidden layers with the same number of neurons 
as the initial hidden layer. Each hidden layer is followed by a ReLU activation function and a dropout layer to 
introduce regularization and prevent overfitting.
'''
class FNNv4(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FNNv4, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, 2*hidden_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(2*hidden_dim, 4*hidden_dim)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(4*hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

"""
Try different activation functions: Instead of using ReLU in all hidden layers, you can experiment with different
activation functions such as sigmoid or tanh. Different activation functions may help the model learn different types
of patterns and improve its representation capabilities.
"""
class FNNv5(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FNNv5, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation1 = nn.Sigmoid()  # Use Sigmoid activation function
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.activation2 = nn.Tanh()  # Use Tanh activation function
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(2 * hidden_dim, 4 * hidden_dim)
        self.activation3 = nn.ReLU()  # Use ReLU activation function
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(4 * hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.activation2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.activation3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

"""
Adjust the dropout rate: The dropout layers can help regularize the model and prevent overfitting.
You can try different dropout rates, such as 0.2 or 0.3, and see if it improves the
model's generalization performance.

In this modified model, the dropout rate for the first dropout layer is set to 0.2, while the dropout rates for
the second and third dropout layers are set to 0.3. You can further experiment with different dropout rates
and find the optimal value that improves the model's performance.
"""
class FNNv6(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FNNv6, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)  # Adjust dropout rate to 0.2
        self.fc2 = nn.Linear(hidden_dim, 2*hidden_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)  # Adjust dropout rate to 0.3
        self.fc3 = nn.Linear(2*hidden_dim, 4*hidden_dim)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.3)  # Adjust dropout rate to 0.3
        self.fc4 = nn.Linear(4*hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

"""
High dimensionality to low dimensionality
"""
class FNNv7(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FNNv7, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)  # Adjust dropout rate to 0.2
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)  # Use integer division //
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)  # Adjust dropout rate to 0.3
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)  # Use integer division //
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.3)  # Adjust dropout rate to 0.3
        self.fc4 = nn.Linear(hidden_dim // 4, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

if __name__ == "__main__":

    # Specified Device agnostic
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device}")

    torch.manual_seed(42)

    # Define parameters (If available), Model's hyperparameters, and directory of csv file
    read_csv = pd.read_csv("../dataset/hyperspectral.csv")
    input_dim = int(len(read_csv.iloc[1, 1:-1]))
    print(f"Input Dimiension of the dataset: {input_dim}")
    hidden_dim = 256
    output_dim = 2

    #Create an instance of the model
    model = FNNv7(input_dim, hidden_dim, output_dim)

    # Print the model summary
    summary(model, (input_dim,))


    '''
    /// Important Information About The Above Code ///
    '''

    """
    The comma (,) after input_dim is used to create a tuple with a single element. In Python, when creating a tuple
    with a single element, you need to include a comma after the element to differentiate it from just a regular value
    enclosed in parentheses.
    """

    """
    The choice of the hidden dimension (hidden_dim) depends on the complexity of the problem and the size of the input 
    data. It is not necessary for the hidden dimension to match the input dimension (input_dim).

    In your example, if input_dim is 204, and you choose hidden_dim as 64, it means that you are reducing the 
    dimensionality of the data from 204 to 64 in the hidden layer. This reduction can help in learning more compact 
    representations and potentially reduce overfitting. However, it also means that you are compressing the 
    information, and some information might be lost in the process.

    The selection of the appropriate hidden dimension is often based on empirical experimentation and tuning. 
    It is recommended to start with a smaller hidden dimension and gradually increase its size if needed. 
    You can train the model with different hidden dimensions and evaluate its performance using validation metrics 
    such as accuracy or loss. This will help you determine the optimal hidden dimension for your 
    specific task and dataset.
    """