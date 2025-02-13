import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Directories
image_dir = 'Data/Raw_data/images/ISIC-images/'
metadata_path = 'Data/Raw_data/metadata/metadata.csv'

# Load metadata
df = pd.read_csv(metadata_path)

# **Pre-cleaning Data Insights**
print("\n#### Raw Metadata Overview ####")
print(f"Number of observations: {df.shape[0]}")
print(f"Number of features: {df.shape[1]}\n")
print(df.info())  # Data types and missing values
print("\n#### Missing Values Per Column (Raw) ####")
print(df.isnull().sum())
print("\n#### Descriptive Statistics (Raw) ####")
print(df.describe())

# Drop unnecessary columns
df = df.drop(columns=['isic_id', 'lesion_id', 'attribution', 'copyright_license', 'image_type'])

# Process the age variable (discretize into bins)
bins = [0, 20, 40, 60, 80, np.inf]
labels = ['0-20', '21-40', '41-60', '61-80', '80+']
df['age_group'] = pd.cut(df['age_approx'], bins=bins, labels=labels)
df = df.drop(columns=['age_approx'])

# Convert boolean columns
df['concomitant_biopsy'] = df['concomitant_biopsy'].astype(bool)
df['melanocytic'] = df['melanocytic'].astype(bool)

# Encode categorical variables
categorical_cols = ['anatom_site_general', 'benign_malignant', 'diagnosis',
                    'diagnosis_1', 'diagnosis_2', 'diagnosis_3',
                    'diagnosis_confirm_type', 'sex', 'age_group']
le = LabelEncoder()
for col in categorical_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

# Handle missing values
df.fillna('Unknown', inplace=True)

# **Post-cleaning Data Insights**
print("\n#### Cleaned Metadata Overview ####")
print(f"Number of observations: {df.shape[0]}")
print(f"Number of features: {df.shape[1]}\n")
print(df.info())  # Data types and non-null count
print("\n#### Missing Values Per Column (Cleaned) ####")
print(df.isnull().sum())
print("\n#### Descriptive Statistics (Cleaned) ####")
print(df.describe())

# Save cleaned metadata
output_metadata_path = "Data/Processed_data/metadata_cleaned_final.csv"
df.to_csv(output_metadata_path, index=False)
print(f"\nFinal cleaned metadata saved to {output_metadata_path}")
