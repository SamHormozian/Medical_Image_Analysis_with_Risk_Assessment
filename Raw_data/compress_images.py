import os
from PIL import Image

# Directories
input_dir = "Raw_data/images/ISIC-images/"
output_dir = "Raw_data/images/ISIC-images/"
os.makedirs(output_dir, exist_ok=True)

# Resize parameters
new_size = (128, 128)  # Target resolution

# Process images
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # Open and resize image
        with Image.open(input_path) as img:
            img_resized = img.resize(new_size)
            img_resized.save(output_path, "JPEG")

        print(f"Resized and saved: {filename}")

print("All images resized!")
