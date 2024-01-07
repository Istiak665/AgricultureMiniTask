import pandas as pd

# Read csv file
df = pd.read_csv("../dataset/RGB_imaging.csv")
print(df.head())

# Check missing values entire dataset
missing_values = df.isnull().sum()
print(missing_values)

# Check for missing values in each column
missing_values_per_column = df.isnull().sum(axis=0)
print(missing_values_per_column)

# Check for missing values in each row
missing_values_per_row = df.isnull().sum(axis=1)
print(missing_values_per_row)

x=4