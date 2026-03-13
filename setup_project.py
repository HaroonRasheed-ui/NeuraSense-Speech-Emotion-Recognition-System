"""
Run this once to set up the project properly:
  python setup_project.py
"""
import subprocess
import sys
import os


def run(cmd):
    print(f"  >> {cmd}")
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.returncode != 0 and result.stderr:
        print(f"  ⚠️  {result.stderr.strip()}")
    return result.returncode == 0


def main():
    print("=" * 55)
    print("  EMOTION RECOGNITION — PROJECT SETUP")
    print("=" * 55)

    # 1. Check Python version
    v = sys.version_info
    print(f"\n✅ Python {v.major}.{v.minor}.{v.micro}")
    if v.major < 3 or v.minor < 8:
        print("❌ Python 3.8+ required!")
        sys.exit(1)

    # 2. Upgrade pip
    print("\n📦 Step 1: Upgrading pip...")
    run(f"{sys.executable} -m pip install --upgrade pip")

    # 3. Install all dependencies
    print("\n📦 Step 2: Installing dependencies...")
    packages = [
        "tensorflow",
        "librosa",
        "numpy",
        "scikit-learn",
        "flask",
        "flask-socketio",
        "sounddevice",
        "scipy",
        "matplotlib",
        "seaborn",
        "pandas",
        "tqdm",
        "eventlet",
        "python-socketio",
        "kaggle",
        "soundfile",
    ]

    # Try pyaudio (can fail on Windows without build tools)
    print("\n  Installing PyAudio (Windows may need extra step)...")
    result = run(f"{sys.executable} -m pip install pyaudio")
    if not result:
        print("  ⚠️  PyAudio failed. Trying pipwin method...")
        run(f"{sys.executable} -m pip install pipwin")
        run(f"{sys.executable} -m pipwin install pyaudio")

    for pkg in packages:
        ok = run(f"{sys.executable} -m pip install {pkg}")
        status = "✅" if ok else "⚠️ "
        print(f"  {status} {pkg}")

    # 4. Verify critical imports
    print("\n🔍 Step 3: Verifying imports...")
    critical = [
        ("tensorflow", "import tensorflow as tf; print('  TF:', tf.__version__)"),
        ("librosa",    "import librosa; print('  librosa:', librosa.__version__)"),
        ("flask",      "from flask import Flask; print('  Flask: OK')"),
        ("sklearn",    "import sklearn; print('  sklearn:', sklearn.__version__)"),
        ("numpy",      "import numpy as np; print('  numpy:', np.__version__)"),
    ]

    all_ok = True
    for name, check in critical:
        r = subprocess.run([sys.executable, "-c", check],
                           capture_output=True, text=True)
        if r.returncode == 0:
            print(f"  ✅ {r.stdout.strip()}")
        else:
            print(f"  ❌ {name} FAILED — {r.stderr.strip()[:80]}")
            all_ok = False

    # 5. Create necessary folders
    print("\n📁 Step 4: Creating project folders...")
    for folder in ['models', 'dataset', 'dataset/ravdess', 'dataset/tess']:
        os.makedirs(folder, exist_ok=True)
        print(f"  ✅ ./{folder}/")

    # 6. Check utils package
    print("\n🔧 Step 5: Checking utils package...")
    init_file = os.path.join(os.path.dirname(__file__), 'utils', '__init__.py')
    if os.path.exists(init_file):
        print("  ✅ utils/__init__.py exists")
    else:
        os.makedirs('utils', exist_ok=True)
        with open(init_file, 'w') as f:
            f.write("# Utils package\n")
        print("  ✅ utils/__init__.py created")

    # Done
    print("\n" + "=" * 55)
    if all_ok:
        print("  🎉 Setup Complete! All dependencies installed.")
        print("\n  Next steps:")
        print("  1. python download_dataset.py  ← Download data")
        print("  2. python train_model.py        ← Train model (~30-90 min)")
        print("  3. python app.py               ← Launch web app")
    else:
        print("  ⚠️  Setup done but some packages failed.")
        print("  Try running:  pip install -r requirements.txt")
    print("=" * 55)


if __name__ == '__main__':
    main()