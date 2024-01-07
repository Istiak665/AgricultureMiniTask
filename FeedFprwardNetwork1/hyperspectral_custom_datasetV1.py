import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Define a custom dataset
class CustomDatasetV1(Dataset):
    def __init__(self, csv_file):
        self.data = []
        self.labels = []

        # Load the data from the CSV file
        with open(csv_file, 'r') as f:
            all_lines = f.readlines()
            for each_line in all_lines[1:]:
                each_line = each_line.strip().split(',')
                features = list(map(float, each_line[1:-1])) # Extract Features
                target = each_line[-1] # Extract target columns
                self.data.append(features)
                self.labels.append(target)

        # Use LabelEncoder to encode categorical labels
        label_encoder = LabelEncoder()
        self.labels = label_encoder.fit_transform(self.labels)

        # Apply feature normalization using StandardScaler
        scalerA = StandardScaler()
        self.data = scalerA.fit_transform(self.data)

        # Apply feature normalization using MinMaxScaler
        # scalerB = MinMaxScaler()
        # self.data = scalerB.fit_transform(self.data)

    def __getitem__(self, index):
        x = torch.tensor(self.data[index])
        y = torch.tensor(self.labels[index])
        return x, y

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    # Define model parameters (If available) and directory of csv file
    CSV_FILE = "../dataset/hyperspectral.csv"

    # Specified device agnostic
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device}")

    torch.manual_seed(42)

    # Create an instance of custom dataset
    HyperSpectral_Dataset = CustomDatasetV1(CSV_FILE)

    # Accessing features and labels for the first sample
    features, label = HyperSpectral_Dataset[2]
    print("Features:", features)
    print(features.shape)
    print("Label:", label)



