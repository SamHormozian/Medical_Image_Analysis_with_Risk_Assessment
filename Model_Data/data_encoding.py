import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Load the cleaned metadata
metadata_path = "Data/Processed_data/metadata_cleaned_final.csv"
metadata = pd.read_csv(metadata_path)

print("Loaded Metadata:")
print(metadata.head())

# Identify non-numeric columns
print("\nData Types:")
print(metadata.dtypes)

non_numeric_columns = metadata.select_dtypes(include=['object', 'bool']).columns
print(f"\nNon-numeric columns: {list(non_numeric_columns)}")

# Encode binary columns (e.g., True/False)
if "melanocytic" in metadata.columns:
    metadata["melanocytic"] = metadata["melanocytic"].astype(int)  # Convert True/False to 1/0

# Encode categorical columns using one-hot encoding
categorical_columns = ["sex", "anatom_site_general", "diagnosis_confirm_type"]
metadata = pd.get_dummies(metadata, columns=categorical_columns, drop_first=True)

# Normalize numerical columns (e.g., age_approx)
if "age_approx" in metadata.columns:
    scaler = StandardScaler()
    metadata["age_approx_scaled"] = scaler.fit_transform(metadata["age_approx"].values.reshape(-1, 1))
    metadata = metadata.drop(columns=["age_approx"])  # Drop original column if scaled version is used

# Drop irrelevant or identifier columns
metadata = metadata.drop(columns=["lesion_id", "isic_id"], errors="ignore")

# Verify processed data
print("\nProcessed Metadata Info:")
print(metadata.info())
print(metadata.head())

# Save the encoded dataset
output_path = "Model_Data/metadata_encoded.csv"
metadata.to_csv(output_path, index=False)
print(f"\nEncoded metadata saved to {output_path}")
