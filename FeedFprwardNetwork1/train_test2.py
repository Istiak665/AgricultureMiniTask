import torch
from torch import nn
from hyperspectral_custom_datasetV2 import CustomDatasetV2
from FeedForwardNet import FNNv1, FNNv2, FNNv3, FNNv4, FNNv5, FNNv6, FNNv7
import pandas as pd
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

def train_loop(model, train_dataloader, loss_function, optimizer, device):
    model.train()
    #Initialize variables for loss and accuracy
    train_loss = 0.0
    train_correct_predictions = 0
    train_total_predictions = 0

    # Iterate over the training data batches:
    for batches, (inputs, labels) in enumerate(train_dataloader):
        # Move the data to the appropriate device:
        inputs = inputs.to(device).float() # Convert input features to float data type when we use MinMax Scaler
        labels = labels.to(device).long()

        # Perform the forward pass and calculate the loss
        outputs = model(inputs)
        loss = loss_function(outputs, labels)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate training loss
        train_loss += loss.item() # This line adds the current batch's loss to the cumulative training loss.

        # Calculate training accuracy
        _, predicted_labels = torch.max(outputs, 1)
        train_correct_predictions += (predicted_labels == labels).sum().item()
        train_total_predictions += labels.size(0)

    # Calculate average training loss
    train_loss /= len(train_dataloader)

    return train_loss, train_correct_predictions, train_total_predictions

def test_loop(model, test_dataloader, loss_function, device):
    model.eval()  # Set the model to evaluation mode

    test_loss = 0.0
    test_correct_predictions = 0
    test_total_predictions = 0

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            # inputs = inputs.to(device)
            inputs = inputs.to(device).float()  # Convert input features to float data type when we use MinMax Scaler
            labels = labels.to(device).long()

            # Forward pass
            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            # Calculate test loss
            test_loss += loss.item()

            # Calculate test accuracy
            _, predicted_labels = torch.max(outputs, 1)
            test_correct_predictions += (predicted_labels == labels).sum().item()
            test_total_predictions += labels.size(0)

    # Calculate average test loss
    test_loss /= len(test_dataloader)

    return test_loss, test_correct_predictions, test_total_predictions

if __name__ == "__main__":
    # Specified Device agnostic
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device}")

    torch.manual_seed(42)
    # Define model parameters (If available) and directory of csv file
    CSV_FILE = "../dataset/hyperspectral.csv"
    # Define parameters (If available), Model's hyperparameters, and directory of csv file
    read_csv = pd.read_csv("../dataset/hyperspectral.csv")
    input_dim = int(len(read_csv.iloc[1, 1:-1]))
    print(f"Input Dimiension of the dataset: {input_dim}")
    # input_dim = 204
    hidden_dim = 512
    output_dim = 2

    # Create Dataset Instance and Make dataloader for train and test
    model_dataset = CustomDatasetV2(CSV_FILE)
    print(model_dataset)

    # Split for Train and Test
    # Define the split ratios
    train_ratio = 0.8
    test_ratio = 1 - train_ratio
    # Calculate the number of samples for each subset
    train_size = int(train_ratio * len(model_dataset))
    test_size = len(model_dataset) - train_size

    # Split the dataset into training and testing subsets
    train_dataset, test_dataset = random_split(model_dataset, [train_size, test_size])

    # Create dataloaders for training and testing
    batch_size = 8 #(8, 16, 32, 64, 128)  # Set your desired batch size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Let's check
    # Iterate over the train_dataloader and print the dataset content
    """
    // --- Below code will check and print dataloader --- // 
    """
    '''
    for batch_idx, (inputs, labels) in enumerate(train_dataloader):
        print(f"Batch {batch_idx + 1}:")
        print("Inputs:", inputs)
        print("Labels:", labels)
        print("--------------------")
    '''

    # Let's create model Instance
    my_modelV1 = FNNv7(input_dim, hidden_dim, output_dim).to(device)
    print(my_modelV1)

    # Set training parameters
    num_epochs = 200
    learning_rate = 0.01
    # Define loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(my_modelV1.parameters(), lr=learning_rate)

    # Run the Train and Test Loop

    # Define the variables to stoire values during training of the model
    train_losses = []
    train_acc = []
    test_losses = []
    test_acc = []

    """
    // ---------- Let's Run----------//
    """
    for epoch in range(num_epochs):
        train_loss, train_correct_predictions, train_total_predictions = train_loop(
            model = my_modelV1,
            train_dataloader = train_dataloader,
            loss_function = loss_function,
            optimizer = optimizer,
            device = device
        )

        test_loss, test_correct_predictions, test_total_predictions = test_loop(
            model = my_modelV1,
            test_dataloader = test_dataloader,
            loss_function = loss_function,
            device = device
        )

        # Store train and test losses values into the variables
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        # Store accuarcy values into the accuracy variables
        train_accuracy = 100.0 * train_correct_predictions / train_total_predictions
        test_accuracy = 100.0 * test_correct_predictions / test_total_predictions

        train_acc.append(train_accuracy)
        test_acc.append(test_accuracy)


        # Print train and test loss, accuracy every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch: {epoch + 1}/{num_epochs}")
            print("==========")
            print(f"Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.2f}%")
            print(f"Test Loss : {test_loss:.4f}  - Test Accuracy : {test_accuracy:.2f}%")


    """
    // ---------- Plot the training and testing loss and accuracy----------//
    """
    # Plot the training and testing loss and accuracy
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.legend()
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    # Accuracy
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(test_acc, label='Test Accuracy')
    plt.legend()
    plt.title('Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()