import os
import pandas as pd

#Directories
image_dir = 'Data/Raw_data/images/ISIC-images/'
dataset_dir = 'Data\Raw_data\metadata\metadata.csv'

#image analysis
image_files = os.listdir(image_dir)
print(f"Number of images: {len(image_files)}")
print(f"Sample image filenames: {image_files[:5]}")

#metadata analysis
metadata = pd.read_csv(dataset_dir)
print(f"Metadata shape: {metadata.shape}")
print("Metadata columns:")
print(metadata.columns)
print("Sample rows:")
print(metadata.head())
