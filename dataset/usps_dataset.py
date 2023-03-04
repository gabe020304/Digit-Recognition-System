from sklearn.datasets import fetch_openml

# Load the USPS dataset
usps = fetch_openml('usps', version=2, parser='auto')

# Print the shape of the dataset
print(f"USPS data shape: {usps.data.shape}")
print(f"USPS target shape: {usps.target.shape}")
