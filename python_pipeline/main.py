import os
import sys
import shutil
import urllib.request
import tarfile
import random
import numpy as np
from PIL import Image, ImageFilter
import math

import config as cfg
import train_model

def simulate_esp32_camera(img):
    # resize
    img = img.resize((cfg.IMG_WIDTH, cfg.IMG_HEIGHT), Image.Resampling.BILINEAR)
    img_arr = np.array(img, dtype=np.float32)

    # Slight overall contrast compression (sensor gamma)
    img_arr = np.clip(img_arr * 0.95 + 8, 0, 255)

    img = Image.fromarray(img_arr.astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    img_arr = np.array(img, dtype=np.float32)

    # Gaussian sensor noise only
    noise = np.random.normal(0, 5.0, img_arr.shape)
    img_arr = np.clip(img_arr + noise, 0, 255)

    # RGB565 bit depth reduction
    img_arr[:, :, 0] = (img_arr[:, :, 0].astype(np.uint8) >> 3) << 3
    img_arr[:, :, 1] = (img_arr[:, :, 1].astype(np.uint8) >> 2) << 2
    img_arr[:, :, 2] = (img_arr[:, :, 2].astype(np.uint8) >> 3) << 3

    return Image.fromarray(img_arr.astype(np.uint8))

def empty_directory_contents(dir_path):
    if os.path.exists(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path): os.unlink(file_path)
                elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(f"[ERROR] Failed to delete {file_path}. Reason: {e}")

def clear_directories():
    print("=== Clearing Old Data ===")
    if os.path.exists(cfg.CLEAN_DATA_DIR):
        empty_directory_contents(cfg.CLEAN_DATA_DIR)
    for class_id in cfg.FOLDERS_TO_CLEAR:
        target_dir = os.path.join(cfg.RAW_DATA_DIR, class_id)
        if os.path.exists(target_dir):
            empty_directory_contents(target_dir)
    print("====================================\n")

def create_directory_structure():
    print("=== Creating Directory Structure ===")
    for class_id in cfg.CLASSES:
        os.makedirs(os.path.join(cfg.RAW_DATA_DIR, class_id), exist_ok=True)
    for split in cfg.SPLITS:
        for class_id in cfg.CLASSES:
            os.makedirs(os.path.join(cfg.CLEAN_DATA_DIR, split, class_id), exist_ok=True)
    print(f"[OK] Directory structure verified.")
    print("====================================\n")
    
def copy_user_images():
    """Copy user images from data/myimg directly into raw_data."""
    print("=== Copying User Images ===")
    myimg_dir = os.path.join(cfg.DATA_DIR, "myimg")
    
    if not os.path.exists(myimg_dir):
        print(f"[INFO] No user image folder found at {myimg_dir}, skipping.")
        print("====================================\n")
        return

    total_copied = 0
    for class_id in cfg.CLASSES:
        src_dir = os.path.join(myimg_dir, class_id)
        dst_dir = os.path.join(cfg.RAW_DATA_DIR, class_id)

        if not os.path.exists(src_dir):
            print(f"[INFO] No user images for class {class_id}, skipping.")
            continue

        files = [f for f in os.listdir(src_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not files:
            print(f"[INFO] Class {class_id} user folder is empty, skipping.")
            continue

        os.makedirs(dst_dir, exist_ok=True)
        copied = 0
        for f in files:
            src_path = os.path.join(src_dir, f)
            dst_path = os.path.join(dst_dir, f"user_{class_id}_{f}")
            try:
                shutil.copy2(src_path, dst_path)
                copied += 1
            except Exception as e:
                print(f"[WARNING] Could not copy {f}: {e}")

        print(f"[OK] Class {class_id}: copied {copied} user images -> {dst_dir}")
        total_copied += copied

    print(f"[OK] Total user images copied: {total_copied}")
    print("====================================\n")
    
def download_progress_hook(count, block_size, total_size):
    if total_size > 0:
        percent = int((count * block_size / total_size) * 100)
        print(f"       ... {min(percent, 100)}% downloaded", end="\r")
    
def download_data():
    print("=== Downloading Public Datasets ===")
    class2_dir = os.path.join(cfg.RAW_DATA_DIR, "2")
    if len(os.listdir(class2_dir)) < 1000:
        print("[INFO] Downloading LFW (Unknown Faces) for Class 2...")
        try:
            from sklearn.datasets import fetch_lfw_people
            lfw_cropped = fetch_lfw_people(color=True, min_faces_per_person=1)
            lfw_full = fetch_lfw_people(color=True, min_faces_per_person=1, slice_=None)
            
            for i in range(300):
                img = Image.fromarray((lfw_cropped.images[i] * 255).astype('uint8'))
                simulate_esp32_camera(img).save(os.path.join(class2_dir, f"lfw_cropped_{i:04d}.jpg"))
            for i in range(300, 600):
                img = Image.fromarray((lfw_full.images[i] * 255).astype('uint8'))
                simulate_esp32_camera(img).save(os.path.join(class2_dir, f"lfw_fullbg_{i:04d}.jpg"))
            print("[OK] Class 2 Download Complete.\n")
        except ImportError:
            print("[ERROR] scikit-learn or Pillow not installed.")
    else:
        print("[INFO] Class 2 already downloaded. Skipping.\n")

    class3_dir = os.path.join(cfg.RAW_DATA_DIR, "3")
    if len(os.listdir(class3_dir)) < 400:
        print("[INFO] Processing MIT Indoor Scenes for Class 3 Backgrounds...")
        tar_path = os.path.join(cfg.DATA_DIR, "indoorCVPR_09.tar")
        if not os.path.exists(tar_path):
            print("       Downloading MIT tar file (~2.4 GB)...")
            try:
                urllib.request.urlretrieve("http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar", tar_path, reporthook=download_progress_hook)
            except Exception as e:
                print(f"\n[ERROR] Download failed: {e}")
                sys.exit(1)

        valid_categories =["office", "corridor", "bedroom", "classroom", "auditorium", "dining_room", "garage", "kitchen", "livingroom", "meeting_room", "waiting_room"]
        target_per_category = math.ceil(400 / len(valid_categories))
        category_counts = {cat: 0 for cat in valid_categories}
        total_extracted = 0

        try:
            with tarfile.open(tar_path, 'r') as tar_ref:
                for member in tar_ref:
                    if total_extracted >= (target_per_category * len(valid_categories)): break
                    if member.isfile() and member.name.endswith(".jpg"):
                        for cat in valid_categories:
                            if f"Images/{cat}/" in member.name and category_counts[cat] < target_per_category:
                                f = tar_ref.extractfile(member)
                                if f is not None:
                                    img = simulate_esp32_camera(Image.open(f).convert("RGB"))
                                    img.save(os.path.join(class3_dir, f"mit_indoor_{cat}_{category_counts[cat]:02d}.jpg"))
                                    category_counts[cat] += 1
                                    total_extracted += 1
                                break
        except tarfile.ReadError:
            print("\n[ERROR] Corrupted .tar file. Deleting. Restart script to redownload.")
            os.remove(tar_path)
            sys.exit(1)
        print("\n[OK] Class 3 Extraction Complete.")
    else:
        print("[INFO] Class 3 already downloaded. Skipping.")
    print("====================================\n")

def split_data():
    print("=== Splitting Data into Train/Validation/Test ===")
    random.seed(42)

    for class_id in cfg.CLASSES:
        raw_class_dir = os.path.join(cfg.RAW_DATA_DIR, class_id)
        if not os.path.exists(raw_class_dir): continue

        files =[f for f in os.listdir(raw_class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not files: continue

        files.sort()
        random.shuffle(files)

        train_count = int(len(files) * 0.6)
        val_count   = int(len(files) * 0.2)

        train_files = files[:train_count]
        val_files   = files[train_count:train_count + val_count]
        test_files  = files[train_count + val_count:]

        def copy_files(file_list, split_name, cid):
            dst_dir = os.path.join(cfg.CLEAN_DATA_DIR, split_name, cid)
            for idx, f in enumerate(file_list):
                ext = os.path.splitext(f)[1]
                shutil.copy2(os.path.join(cfg.RAW_DATA_DIR, cid, f), os.path.join(dst_dir, f"{idx:05d}{ext}"))

        copy_files(train_files, "train", class_id)
        copy_files(val_files,   "validation", class_id)
        copy_files(test_files,  "test", class_id)
        print(f"[OK] Class {class_id}: {len(files)} images -> {len(train_files)} Train / {len(val_files)} Val / {len(test_files)} Test")

if __name__ == "__main__":
    print("Starting ESP32 Face Recognition Pipeline...\n")
    
    clear_directories()
    create_directory_structure()
    copy_user_images()
    download_data()
    split_data()
    
    train_model.train_and_export()
    
    print("\n[SUCCESS] Pipeline finished! Your ESP32 C++ files are ready.")