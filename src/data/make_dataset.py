import kagglehub
import os
import shutil

def download_data():
    """Downloads the cardiovascular disease dataset from Kaggle."""
    print("Downloading dataset...")
    
    # Download latest version
    path = kagglehub.dataset_download("sulianova/cardiovascular-disease-dataset")
    print("Path to downloaded files:", path)

    # Define target directory
    target_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")
    
    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy files to data/raw
    # Note: kagglehub path is usually a directory containing the files
    if os.path.isdir(path):
        for file_name in os.listdir(path):
            full_file_name = os.path.join(path, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, target_dir)
                print(f"Copied {file_name} to {target_dir}")
    else:
        # Fallback if path is just a file
        shutil.copy(path, target_dir)
        print(f"Copied {path} to {target_dir}")

if __name__ == "__main__":
    download_data()
