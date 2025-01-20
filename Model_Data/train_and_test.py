import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Paths to data files
cleaned_metadata_path = "Data/Processed_data/metadata_cleaned_final.csv"
encoded_metadata_path = "Model_Data/metadata_encoded.csv"
updated_encoded_metadata_path = "Model_Data/metadata_encoded_with_ids.csv"

# Load the datasets
cleaned_metadata = pd.read_csv(cleaned_metadata_path)
encoded_metadata = pd.read_csv(encoded_metadata_path)

# Add `isic_id` from cleaned_metadata to encoded_metadata
encoded_metadata["isic_id"] = cleaned_metadata["isic_id"]

# Save the updated encoded metadata
encoded_metadata.to_csv(updated_encoded_metadata_path, index=False)
print(f"Updated encoded metadata saved to {updated_encoded_metadata_path}")

# Load the updated metadata
metadata = pd.read_csv(updated_encoded_metadata_path)

# Separate features, target, and image IDs
X = metadata.drop(columns=["benign_malignant", "isic_id"])  # Features
y = metadata["benign_malignant"].apply(lambda x: 1 if x == "malignant" else 0)  # Target (0 = benign, 1 = malignant)
image_ids = metadata[["isic_id"]]  # Retain `isic_id` as a DataFrame with column name

# Split the dataset
X_train, X_test, y_train, y_test, image_ids_train, image_ids_test = train_test_split(
    X, y, image_ids, test_size=0.3, stratify=y, random_state=42
)

# Save the splits
X_train.to_csv("Model_Data/Training/X_train.csv", index=False)
X_test.to_csv("Model_Data/Testing/X_test.csv", index=False)
y_train.to_csv("Model_Data/Training/y_train.csv", index=False)
y_test.to_csv("Model_Data/Testing/y_test.csv", index=False)

# Save image IDs with column name intact
image_ids_train.to_csv("Model_Data/Training/image_ids_train.csv", index=False)
image_ids_test.to_csv("Model_Data/Testing/image_ids_test.csv", index=False)

print("Train and test datasets saved successfully!")

# Reload image IDs to validate
image_ids_train = pd.read_csv("Model_Data/Training/image_ids_train.csv")
image_ids_test = pd.read_csv("Model_Data/Testing/image_ids_test.csv")

# Validate Class Distribution
print("\nClass Distribution in Train Set:")
print(y_train.value_counts())

print("\nClass Distribution in Test Set:")
print(y_test.value_counts())

# Validate Image IDs
image_dir = "Data/Raw_data/images/ISIC-images/"

# Check if all train image IDs exist
invalid_train_ids = [img for img in image_ids_train["isic_id"] if not os.path.exists(os.path.join(image_dir, img))]
print(f"\nInvalid train image IDs: {invalid_train_ids}")

# Check if all test image IDs exist
invalid_test_ids = [img for img in image_ids_test["isic_id"] if not os.path.exists(os.path.join(image_dir, img))]
print(f"Invalid test image IDs: {invalid_test_ids}")

# Sample Data Inspection
print("\nSample X_train:")
print(X_train.head())

print("\nSample y_train:")
print(y_train.head())

print("\nSample image_ids_train:")
print(image_ids_train.head())

# Validate shapes
print("\nShapes of the Data Splits:")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"image_ids_train shape: {image_ids_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"image_ids_test shape: {image_ids_test.shape}")
