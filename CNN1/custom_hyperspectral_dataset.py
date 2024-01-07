# Import Necessary Libraries
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# Create a class of Custom Dataset of Hyperspectral Seed for Germinition
class Hspectral_Dtatset(Dataset):
    def __init__(self, csv_file):
        # Read csv file
        df = pd.read_csv(csv_file)
        # Extract features and labels
        self.data = df.iloc[:, 1:-1].values
        self.labels = df.iloc[:, -1].values
        # Use LabelEncoder to encode categorical labels
        label_encoder_object = LabelEncoder()
        self.labels = label_encoder_object.fit_transform(self.labels)
        # Apply MinMax Scaler to Normalize the Input Data
        scaler = MinMaxScaler()
        self.data = scaler.fit_transform(self.data)

    def __getitem__(self, index):
        x = torch.tensor(self.data[index]).float()
        y = torch.tensor(self.labels[index]).float()
        return x, y

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    # Define dataset path
    CSV_FILE = "../dataset/hyperspectral.csv"
    # Define Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Create an Instance of Custom dataset and chcek the results
    dataset_customV1 = Hspectral_Dtatset(CSV_FILE)
    features, label = dataset_customV1[0]

    print(f"Features: {features} and Feature Shape: {features.shape}")
    print("Label:", label)

    a =12
