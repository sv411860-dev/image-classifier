import os
import zipfile
import urllib.request
import sys

DATA_URL = "https://download.mlcc.google.com/mledu-datasets/cats_and_dogs_filtered.zip"
TARGET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ZIP_PATH = os.path.join(TARGET_DIR, "cats_and_dogs_filtered.zip")
EXTRACTED_DIR = os.path.join(TARGET_DIR, "cats_and_dogs_filtered")

def download_and_extract():
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    if os.path.exists(EXTRACTED_DIR):
        print(f"Dataset already exists at: {EXTRACTED_DIR}")
        print_dataset_summary(EXTRACTED_DIR)
        return EXTRACTED_DIR

    print(f"Downloading Cats & Dogs dataset from:\n  {DATA_URL}")

    req = urllib.request.Request(DATA_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(ZIP_PATH, 'wb') as out_file:
        total_size = int(response.info().get('Content-Length', -1))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB chunks

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = downloaded * 100 / total_size
                sys.stdout.write(f"\rDownloading: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent:.1f}%)")
            else:
                sys.stdout.write(f"\rDownloading: {downloaded / (1024*1024):.1f} MB")
            sys.stdout.flush()

    print("\nDownload complete! Extracting files...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(TARGET_DIR)

    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

    print("Dataset extracted successfully!")
    print_dataset_summary(EXTRACTED_DIR)
    return EXTRACTED_DIR

def print_dataset_summary(base_dir):
    for split in ["train", "validation"]:
        split_path = os.path.join(base_dir, split)
        if os.path.exists(split_path):
            cats = len(os.listdir(os.path.join(split_path, "cats")))
            dogs = len(os.listdir(os.path.join(split_path, "dogs")))
            print(f"  • {split.capitalize()}: {cats} cats, {dogs} dogs (Total: {cats + dogs})")

if __name__ == "__main__":
    download_and_extract()
