#loads and splits pre-cleaned dataset as pandas dataframe, and splits into train and test fractions
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from constants import test_size

# Get the directory of the current script
current_dir = os.path.dirname(__file__)
file_name = 'thyroid_cleaned_num.csv'
file_path = os.path.join(current_dir, file_name)

df = pd.read_csv(file_path)

#inputs and outputs
target_class = 'target_encoded'
column_names = df.columns.tolist()
column_names.remove(target_class)
no_classes = df[target_class].max() + 1

# Feature columns (X) and target column (y)
X = df.drop(columns=['target_encoded']).values
y = df['target_encoded'].values

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Step 2: Create Dataset and DataLoader
class ThyroidDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 1#model can only handle one input at a time 
train_dataset = ThyroidDataset(X_train, y_train)
test_dataset = ThyroidDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)