import os
import pandas as pd
import matplotlib.pyplot as plt

# Directories
image_dir = 'Data/Raw_data/images/ISIC-images/'
metadata_path = 'Data/Raw_data/metadata/metadata.csv'

# Load images and metadata
image_files = os.listdir(image_dir)
print(f"Number of images: {len(image_files)}")
print(f"Sample image filenames: {image_files[:5]}")

metadata = pd.read_csv(metadata_path)
print(f"Metadata shape: {metadata.shape}")
print("Metadata columns:")
print(metadata.columns)
print("Sample rows:")
print(metadata.head())

# Filter valid image files
valid_images = [f for f in image_files if f.endswith('.jpg')]
print(f"Valid images: {len(valid_images)}")

# Ensure alignment by appending '.jpg' to 'isic_id' in metadata
metadata["isic_id"] = metadata["isic_id"].apply(lambda x: f"{x}.jpg")

# Align metadata with valid images
metadata = metadata[metadata["isic_id"].isin(valid_images)]
print(f"Aligned metadata shape: {metadata.shape}")

# Check for mismatched IDs
mismatched_ids = set(metadata["isic_id"]) - set(valid_images)
print(f"Mismatched IDs: {list(mismatched_ids)[:5]}")

# Handle missing values in 'age_approx'
metadata["age_approx"] = metadata["age_approx"].fillna(metadata["age_approx"].median())

# Filter valid labels in 'benign_malignant'
valid_labels = ["benign", "malignant"]
metadata = metadata[metadata["benign_malignant"].isin(valid_labels)]
print(f"Filtered metadata shape (after valid labels): {metadata.shape}")

# Analyze class distribution
class_counts = metadata["benign_malignant"].value_counts()
print("Class distribution:")
print(class_counts)


# Additional cleaning steps

# Drop irrelevant columns
metadata = metadata.drop(columns=["anatom_site_special", "attribution", "copyright_license"], errors="ignore")

# Impute missing values
metadata["anatom_site_general"] = metadata["anatom_site_general"].fillna(metadata["anatom_site_general"].mode()[0])
metadata["diagnosis_confirm_type"] = metadata["diagnosis_confirm_type"].fillna("Unknown")
metadata["sex"] = metadata["sex"].fillna("Unknown")

# Check for remaining missing values
print("Remaining missing values:")
print(metadata.isnull().sum())

# Save cleaned metadata
output_metadata_path = "Data/Processed_data/metadata_cleaned_final.csv"
metadata.to_csv(output_metadata_path, index=False)
print(f"Final cleaned metadata saved to {output_metadata_path}")
