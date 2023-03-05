from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load the USPS dataset
usps = fetch_openml('usps', version=2, parser='auto')

# Preprocess the data
X = np.array(usps.data).reshape(-1, 16, 16, 1).astype('float32') / 255.0
y = to_categorical(usps.target.astype('int'))

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the datasets
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_val shape: {y_val.shape}")
