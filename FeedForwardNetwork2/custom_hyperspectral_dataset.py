# Import Necessary Libraries
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

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


"""
The map() function in Python is a built-in function that applies a given function to each item of an
iterable (e.g., a list, tuple, or string) and returns an iterator that yields the results. 
The general syntax: map(function, iterable)
The function parameter is the function that you want to apply to each item of the iterable.
It can be a built-in function or a custom-defined function. The iterable parameter is the sequence
or collection of items that you want to apply the function to.
"""

if __name__ == "__main__":
    CSV_FILE = "../dataset/hyperspectral.csv"

    # Define Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create an Instance of Custom dataset and chcek the results
    dataset_customV1 = Hspectral_Dtatset(CSV_FILE)

    features, label = dataset_customV1[0]

    print(f"Features: {features} and Feature Shape: {features.shape}")
    print("Label:", label)
