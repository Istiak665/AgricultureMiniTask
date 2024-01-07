# Import Necessary Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# Create a class of Custom Dataset of Hyperspectral Seed for Germinition
class Hspectral_Dtatset(Dataset):
    def __init__(self, csv_file):
        self.data = []
        self.labels = []

        with open(csv_file, 'r') as f:
            all_lines = f.readlines()
            for each_line in all_lines[1:]:
                each_line = each_line.strip().split(',')
                features = list(map(float, each_line[1:-1]))
                target = each_line[-1]

                self.data.append(features)
                self.labels.append(target)


        # Use LabelEncoder to encode categorical labels
        label_encoder_object = LabelEncoder()
        self.labels = label_encoder_object.fit_transform(self.labels)

        # Apply MinMax Scaler to Normalize the Input Data
        scaler = MinMaxScaler()
        self.data = scaler.fit_transform(self.data)

    def __getitem__(self, index):
        x = torch.tensor(self.data[index])
        y = torch.tensor(self.labels[index])
        return x, y

    def __len__(self):
        return len(self.data)

# Create Logistic Regression Class
class Logistic_Regression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Logistic_Regression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x

if __name__ == "__main__":
    CSV_FILE = "../dataset/hyperspectral.csv"
    # Define Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Create an Instance of Custom dataset
    dataset = Hspectral_Dtatset(CSV_FILE)
    # Split datset for train and test
    train_data, test_data, train_labels, test_labels = train_test_split(dataset.data, dataset.labels, test_size=0.2,
                                                                        random_state=42)

    # print(type(train_data))
    # print(len(train_data))
    # print(len(train_labels))
    # print(len(test_data))
    # print(len(test_labels))

    # Convert the data into PyTorch tensors
    train_data = torch.tensor(train_data, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32)

    # Define Input Dimension
    input_dim = train_data.shape[1]
    # print(input_dim)
    output_dim = 1
    # Create Model Instance
    lr_model = Logistic_Regression(input_dim, output_dim)
    # print(lr_model)

    # Train and evaluate the models:
    criterion = nn.BCELoss()
    optimizer = optim.SGD(lr_model.parameters(), lr=0.01)

    num_epochs = 100

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = lr_model(train_data)
        loss = criterion(outputs, train_labels.unsqueeze(1))
        loss.backward()
        optimizer.step()

    # Evaluation
    lr_model.eval()
    with torch.no_grad():
        predictions = (lr_model(test_data) >= 0.5).squeeze().int()
        accuracy = accuracy_score(test_labels, predictions)
        print("Logistic Regression Accuracy:", accuracy)


    """
    // Performance Evaluation and Illustration
    """
    # Import Necessary Libraries
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

    # Calculate ROC curve
    probs = lr_model(test_data).detach().numpy()
    fpr, tpr, thresholds = roc_curve(test_labels, probs)

    # Calculate AUC score
    auc_score = roc_auc_score(test_labels, probs)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='Logistic Regression (AUC = {:.2f})'.format(auc_score))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

    # Calculate confusion matrix
    pred_labels = (probs >= 0.5).astype(int)
    cm = confusion_matrix(test_labels, pred_labels)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    from sklearn.metrics import precision_score, recall_score, f1_score

    # Assuming you have predictions and test_labels as numpy arrays

    # Calculate precision
    precision = precision_score(test_labels, predictions)

    # Calculate recall
    recall = recall_score(test_labels, predictions)

    # Calculate F1-score
    f1 = f1_score(test_labels, predictions)

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
