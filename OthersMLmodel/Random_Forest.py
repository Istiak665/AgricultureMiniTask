# Import Necessary Libraries
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


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
    rf_model = RandomForestClassifier()
    # print(lr_model)

    # Fit the model
    rf_model.fit(train_data, train_labels)


    """
    // Performance Evaluation and Illustration
    """
    # Evaluation
    predictions = rf_model.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)
    print("Random Forest Accuracy:", accuracy)

    """
    // Illustration
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    # Confusion Matrix
    cm = confusion_matrix(test_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Confusion Matrix")
    plt.show()

    # Precision, Recall, F1-score
    from sklearn.metrics import precision_score, recall_score, f1_score

    # Calculate precision
    precision = precision_score(test_labels, predictions)
    # Calculate recall
    recall = recall_score(test_labels, predictions)
    # Calculate F1-score
    f1 = f1_score(test_labels, predictions)

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
