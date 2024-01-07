import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.utils.data import Dataset


# Create a class of Custom Dataset of Hyperspectral Seed for Germination
class Hspectral_Dataset(Dataset):
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

    # Create an Instance of Custom dataset
    dataset = Hspectral_Dataset(CSV_FILE)

    # Split the dataset into train and test sets
    train_data, test_data, train_labels, test_labels = train_test_split(dataset.data, dataset.labels, test_size=0.2, random_state=42)

    # Create Model Instance
    plsda_model = PLSRegression(n_components=2)

    # Fit the model
    plsda_model.fit(train_data, train_labels)

    # Predict using the trained PLS-DA model
    plsda_predictions = plsda_model.predict(test_data)

    # Convert predicted probabilities into class labels using a threshold
    threshold = 0.5
    plsda_predictions_labels = (plsda_predictions >= threshold).astype(int)

    # Evaluate the PLS-DA model
    accuracy = accuracy_score(test_labels, plsda_predictions_labels)
    precision = precision_score(test_labels, plsda_predictions_labels)
    recall = recall_score(test_labels, plsda_predictions_labels)
    f1 = f1_score(test_labels, plsda_predictions_labels)

    print("PLS-DA Accuracy:", accuracy)
    print("PLS-DA Precision:", precision)
    print("PLS-DA Recall:", recall)
    print("PLS-DA F1-score:", f1)

    # Plot the confusion matrix
    cm = confusion_matrix(test_labels, plsda_predictions_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Confusion Matrix")
    plt.show()
