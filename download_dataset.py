"""
Dataset Downloader - Downloads RAVDESS and TESS from Kaggle
Run this FIRST before training: python download_dataset.py
"""

import os
import zipfile
import sys


def check_kaggle():
    """Check if kaggle is configured."""
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")

    if not os.path.exists(kaggle_json):
        print("=" * 60)
        print("  ⚙️  KAGGLE API SETUP REQUIRED")
        print("=" * 60)
        print("\n  Steps to get your Kaggle API key:")
        print("  1. Go to https://www.kaggle.com")
        print("  2. Click your profile → Settings")
        print("  3. Scroll to 'API' section → Click 'Create New Token'")
        print("  4. This downloads 'kaggle.json'")
        print(f"  5. Place it at: {kaggle_json}")
        print("\n  Then run this script again.")
        return False

    # Fix permissions
    os.chmod(kaggle_json, 0o600)
    return True


def download_ravdess():
    """Download RAVDESS dataset from Kaggle."""
    save_path = "./dataset/ravdess"
    os.makedirs(save_path, exist_ok=True)

    if os.path.exists(save_path) and len(os.listdir(save_path)) > 5:
        print("✅ RAVDESS already downloaded, skipping...")
        return True

    print("\n📥 Downloading RAVDESS dataset...")
    print("   Size: ~500MB - Please wait...")

    try:
        import kaggle
        kaggle.api.authenticate()
        os.makedirs("./dataset/ravdess_zip", exist_ok=True)
        kaggle.api.dataset_download_files(
            "uwrfkaggler/ravdess-emotional-speech-audio",
            path="./dataset/ravdess_zip",
            unzip=False
        )

        zip_path = "./dataset/ravdess_zip/ravdess-emotional-speech-audio.zip"
        if os.path.exists(zip_path):
            print("📦 Extracting RAVDESS...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(save_path)
            print("✅ RAVDESS extracted successfully!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def download_tess():
    """Download TESS dataset from Kaggle."""
    save_path = "./dataset/tess"
    os.makedirs(save_path, exist_ok=True)

    if os.path.exists(save_path) and len(os.listdir(save_path)) > 5:
        print("✅ TESS already downloaded, skipping...")
        return True

    print("\n📥 Downloading TESS dataset...")
    print("   Size: ~200MB - Please wait...")

    try:
        import kaggle
        kaggle.api.authenticate()
        os.makedirs("./dataset/tess_zip", exist_ok=True)
        kaggle.api.dataset_download_files(
            "ejlok1/toronto-emotional-speech-set-tess",
            path="./dataset/tess_zip",
            unzip=False
        )

        zip_path = "./dataset/tess_zip/toronto-emotional-speech-set-tess.zip"
        if os.path.exists(zip_path):
            print("📦 Extracting TESS...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(save_path)
            print("✅ TESS extracted successfully!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def manual_instructions():
    """Show manual download instructions."""
    print("\n" + "=" * 60)
    print("  📋 MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("""
If Kaggle download fails, manually download:

  RAVDESS:
  → https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
  → Extract to: ./dataset/ravdess/

  TESS:
  → https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
  → Extract to: ./dataset/tess/

  Then run: python train_model.py
    """)


if __name__ == '__main__':
    print("=" * 60)
    print("  SPEECH EMOTION RECOGNITION - DATASET DOWNLOADER")
    print("=" * 60)

    if not check_kaggle():
        manual_instructions()
        sys.exit(1)

    ravdess_ok = download_ravdess()
    tess_ok = download_tess()

    if ravdess_ok or tess_ok:
        print("\n" + "=" * 60)
        print("  ✅ Datasets ready!")
        print("  Next step: python train_model.py")
        print("=" * 60)
    else:
        manual_instructions()