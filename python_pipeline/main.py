import os
import sys
import shutil
import urllib.request
import tarfile
import random
import numpy as np
from PIL import Image, ImageFilter
import math

# Get the absolute path of the directory this script is in (python_pipeline)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Define the main data paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw_data")
CLEAN_DATA_DIR = os.path.join(DATA_DIR, "clean_dataset")

CLASSES =["0", "1", "2", "3"]
SPLITS = ["train", "validation", "test"]
FOLDERS_TO_CLEAR =["2", "3"]

def simulate_esp32_camera(img):
    """Bridges the Reality Gap by converting high-quality images to ESP32-S3 quality."""
    # 1. Resize directly to 320x240 (avoids cutting off faces)
    img = img.resize((320, 240), Image.Resampling.BILINEAR)
    
    # 2. Add slight blur to simulate cheap plastic lens
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    
    # Convert image to numpy array for math operations
    img_arr = np.array(img, dtype=np.float32)
    
    # 3. Add artificial CMOS sensor noise (grain)
    noise = np.random.normal(loc=0.0, scale=10.0, size=img_arr.shape)
    img_arr = img_arr + noise
    img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
    
    # 4. Simulate RGB565 color depth
    img_arr[:, :, 0] = (img_arr[:, :, 0] // 8) * 8
    img_arr[:, :, 1] = (img_arr[:, :, 1] // 4) * 4
    img_arr[:, :, 2] = (img_arr[:, :, 2] // 8) * 8
    
    return Image.fromarray(img_arr)

def empty_directory_contents(dir_path):
    """Deletes all files and subfolders inside a directory, but keeps the root directory intact."""
    if os.path.exists(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"[ERROR] Failed to delete {file_path}. Reason: {e}")

def clear_directories():
    print("=== Clearing Old Data ===")
    if os.path.exists(CLEAN_DATA_DIR):
        print(f"[INFO] Emptying contents of clean_dataset directory...")
        empty_directory_contents(CLEAN_DATA_DIR)
        
    for class_id in FOLDERS_TO_CLEAR:
        target_dir = os.path.join(RAW_DATA_DIR, class_id)
        if os.path.exists(target_dir):
            print(f"[INFO] Emptying contents of raw data Class {class_id}...")
            empty_directory_contents(target_dir)
    print("====================================\n")

def create_directory_structure():
    print("=== Creating Directory Structure ===")
    for class_id in CLASSES:
        os.makedirs(os.path.join(RAW_DATA_DIR, class_id), exist_ok=True)
    for split in SPLITS:
        for class_id in CLASSES:
            os.makedirs(os.path.join(CLEAN_DATA_DIR, split, class_id), exist_ok=True)
    print(f"[OK] Directory structure verified in {os.path.relpath(DATA_DIR, PROJECT_ROOT)}")
    print("====================================\n")

def download_progress_hook(count, block_size, total_size):
    if total_size > 0:
        downloaded = count * block_size
        percent = int((downloaded / total_size) * 100)
        print(f"       ... {min(percent, 100)}% downloaded", end="\r")

def download_data():
    print("=== Downloading Public Datasets ===")
    
    # ---------------------------------------------------------
    # 1. Download Class 2 (Unknown Faces WITH and WITHOUT Backgrounds)
    # ---------------------------------------------------------
    class2_dir = os.path.join(RAW_DATA_DIR, "2")
    if len(os.listdir(class2_dir)) < 500:
        print("[INFO] Downloading LFW (Unknown Faces) for Class 2...")
        try:
            from sklearn.datasets import fetch_lfw_people
            
            print("       Fetching tightly cropped faces...")
            lfw_cropped = fetch_lfw_people(color=True, min_faces_per_person=1)
            print("       Fetching full images (faces with backgrounds)...")
            lfw_full = fetch_lfw_people(color=True, min_faces_per_person=1, slice_=None)
            
            print(f"[INFO] Converting images to ESP32 quality and saving to {class2_dir}...")
            for i in range(300):
                img_data = (lfw_cropped.images[i] * 255).astype('uint8')
                img = Image.fromarray(img_data)
                img = simulate_esp32_camera(img)
                img.save(os.path.join(class2_dir, f"lfw_cropped_{i:04d}.jpg"))
                
            for i in range(300, 600):
                img_data = (lfw_full.images[i] * 255).astype('uint8')
                img = Image.fromarray(img_data)
                img = simulate_esp32_camera(img)
                img.save(os.path.join(class2_dir, f"lfw_fullbg_{i:04d}.jpg"))
                
            print("[OK] Class 2 Download Complete.\n")
        except ImportError:
            print("[ERROR] scikit-learn or Pillow not installed. Run: pip install scikit-learn Pillow")
    else:
        print(f"[INFO] Class 2 already contains {len(os.listdir(class2_dir))} images. Skipping download.\n")

    # ---------------------------------------------------------
    # 2. Download Class 3 (Backgrounds) - Standard Disk Download
    # ---------------------------------------------------------
    class3_dir = os.path.join(RAW_DATA_DIR, "3")
    if len(os.listdir(class3_dir)) < 400:
        print("[INFO] Processing MIT Indoor Scenes for Class 3 Backgrounds...")
        
        mit_url = "http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar"
        tar_path = os.path.join(DATA_DIR, "indoorCVPR_09.tar")

        # Step A: Download the giant file ONLY if it doesn't already exist
        if not os.path.exists(tar_path):
            print("       Downloading MIT tar file (WARNING: 2.4 GB file, this will take several minutes!)...")
            try:
                urllib.request.urlretrieve(mit_url, tar_path, reporthook=download_progress_hook)
                print("\n       Download complete!")
            except Exception as e:
                print(f"\n[ERROR] Download failed: {e}")
                if os.path.exists(tar_path):
                    os.remove(tar_path)
                sys.exit(1)
        else:
            print("       Found existing MIT tar file. Skipping download!")

        valid_categories =[
            "office", "corridor", "bedroom", "classroom",
            "auditorium", "dining_room", "garage", "kitchen",
            "livingroom", "meeting_room", "waiting_room"
        ]

        target_per_category = math.ceil(400 / len(valid_categories))  # ~37
        target_total = target_per_category * len(valid_categories)
        category_counts = {cat: 0 for cat in valid_categories}
        total_extracted = 0

        print("       Extracting, applying ESP32 filter, and saving to Class 3 folder...")
        
        try:
            with tarfile.open(tar_path, 'r') as tar_ref:
                for member in tar_ref:
                    if total_extracted >= target_total:
                        break

                    if member.isfile() and member.name.endswith(".jpg"):
                        for cat in valid_categories:
                            if f"Images/{cat}/" in member.name and category_counts[cat] < target_per_category:
                                f = tar_ref.extractfile(member)
                                if f is not None:
                                    img = Image.open(f).convert("RGB")
                                    img = simulate_esp32_camera(img)
                                    img.save(os.path.join(class3_dir, f"mit_indoor_{cat}_{category_counts[cat]:02d}.jpg"))
                                    category_counts[cat] += 1
                                    total_extracted += 1

                                    if total_extracted % 20 == 0:
                                        print(f"       ... {total_extracted}/{target_total} images processed", end="\r")
                                break
        except tarfile.ReadError:
            # Catch corruption, delete the bad file, and tell the user to restart
            print("\n[ERROR] The existing .tar file is corrupted or incomplete.")
            print("        Automatically deleting the corrupted file.")
            print("        Please run the script again to re-download.")
            if 'tar_ref' in locals():
                tar_ref.close()
            os.remove(tar_path)
            sys.exit(1)

        print("\n[OK] Class 3 Extraction Complete. (The 2.4 GB tar file was kept in data/)")

    else:
        print(f"[INFO] Class 3 already contains {len(os.listdir(class3_dir))} images. Skipping download.")

    print("====================================\n")

def split_data():
    """Splits raw data into Train (60%), Validation (20%), and Test (20%)."""
    print("=== Splitting Data into Train/Validation/Test ===")
    
    random.seed(42)
    
    for class_id in CLASSES:
        raw_class_dir = os.path.join(RAW_DATA_DIR, class_id)
        
        valid_extensions = ('.jpg', '.jpeg', '.png')
        if not os.path.exists(raw_class_dir):
            continue
            
        files =[f for f in os.listdir(raw_class_dir) if f.lower().endswith(valid_extensions)]
        
        if len(files) == 0:
            print(f"[WARNING] Class {class_id} is currently empty. Skipping split.")
            continue
            
        files.sort()
        random.shuffle(files)
        
        total_files = len(files)
        train_count = int(total_files * 0.6)
        val_count = int(total_files * 0.2)
        
        train_files = files[:train_count]
        val_files = files[train_count:train_count + val_count]
        test_files = files[train_count + val_count:]
        
        def copy_files(file_list, split_name):
            dst_dir = os.path.join(CLEAN_DATA_DIR, split_name, class_id)
            for f in file_list:
                src_path = os.path.join(raw_class_dir, f)
                dst_path = os.path.join(dst_dir, f)
                shutil.copy2(src_path, dst_path)

        copy_files(train_files, "train")
        copy_files(val_files, "validation")
        copy_files(test_files, "test")
        
        print(f"[OK] Class {class_id}: {total_files} images split -> {len(train_files)} Train, {len(val_files)} Val, {len(test_files)} Test.")
        
    print("====================================\n")

def main():
    print("Starting ESP32 Face Recognition Data Pipeline...\n")
    clear_directories()
    create_directory_structure()
    download_data()
    split_data()
    print("Pipeline script finished successfully.")

if __name__ == "__main__":
    main()