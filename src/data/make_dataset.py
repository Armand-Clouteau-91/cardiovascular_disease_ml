import kagglehub
import os
import shutil
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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

def data_cleaning(file_name):
    csv_path = os.path.join("data","raw",f'{file_name}')
    df = pd.read_csv(csv_path)
    df = df.set_index("id")
    df["age"] = df["age"] / 365
    df["age"] =df["age"].astype(int)
    df.dropna()
    
    count = 0
    while ( ((df['ap_hi'] < 50) | (df['ap_hi'] > 250) | (df['ap_lo'] < 30) | (df['ap_lo'] > 160)) .any()) | count < 10:
        df.loc[df['ap_hi'] < 50, 'ap_hi'] *= 10
        df.loc[df['ap_hi'] > 250, 'ap_hi'] //= 10
        df.loc[df['ap_lo'] < 30, 'ap_lo'] *= 10
        df.loc[df['ap_lo'] > 160, 'ap_lo'] //= 10
        count += 1

    invalid_bp = ( (df['ap_hi'] < 50) | (df['ap_hi'] > 250) | (df['ap_lo'] < 30) | (df['ap_lo'] > 160) | (df['ap_hi'] < df['ap_lo']) )
    df = df.drop(df[invalid_bp].index)
    return df

def data_normalization(X_train, X_pred):
    scaler = StandardScaler()
    continuous = ["age", "BMI", "Pulse Pressure"]
    X_train[continuous] = scaler.fit_transform(X_train[continuous])
    X_pred[continuous] = scaler.transform(X_pred[continuous])
    return X_train, X_pred

if __name__ == "__main__":
    download_data()
