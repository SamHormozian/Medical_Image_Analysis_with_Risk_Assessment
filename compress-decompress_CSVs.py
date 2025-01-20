import os
import gzip
import shutil

def compress_csv_files(root_dir):
    """Compress all CSV files in the directory and its subdirectories."""
    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.csv'):
                csv_file = os.path.join(foldername, filename)
                compressed_file = csv_file + '.gz'

                # Skip if already compressed
                if os.path.exists(compressed_file):
                    print(f"Skipping already compressed file: {compressed_file}")
                    continue

                # Compress the file
                with open(csv_file, 'rb') as f_in:
                    with gzip.open(compressed_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                print(f"Compressed: {csv_file} -> {compressed_file}")
                os.remove(csv_file)  # Remove the original CSV file

def decompress_csv_files(root_dir):
    """Decompress all gzipped CSV files in the directory and its subdirectories."""
    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.csv.gz'):
                compressed_file = os.path.join(foldername, filename)
                decompressed_file = compressed_file[:-3]  # Remove the .gz extension

                # Skip if already decompressed
                if os.path.exists(decompressed_file):
                    print(f"Skipping already decompressed file: {decompressed_file}")
                    continue

                # Decompress the file
                with gzip.open(compressed_file, 'rb') as f_in:
                    with open(decompressed_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                print(f"Decompressed: {compressed_file} -> {decompressed_file}")
                os.remove(compressed_file)  # Remove the compressed file

def main():
    print("Select an option:")
    print("1. Compress all CSV files")
    print("2. Decompress all gzipped CSV files")
    choice = input("Enter your choice (1/2): ")

    root_dir = os.getcwd()  # Root directory is the current working directory

    if choice == '1':
        compress_csv_files(root_dir)
    elif choice == '2':
        decompress_csv_files(root_dir)
    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == '__main__':
    main()
