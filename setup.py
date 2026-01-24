import os
import sys
import subprocess
import platform
import importlib.util

# Configuration
VENV_NAME = "venv"
REQUIREMENTS_FILE = "requirements.txt"
# This URL forces the script to get the GPU version of PyTorch
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu124"


def install_virtualenv_if_missing():
    """Checks if 'virtualenv' library exists. If not, installs it."""
    if importlib.util.find_spec("virtualenv") is None:
        print("🔧 'virtualenv' library not found. Installing it now...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "virtualenv"]
            )
            print("✅ 'virtualenv' library installed.")
        except subprocess.CalledProcessError:
            print(
                "❌ Failed to install 'virtualenv'. Please run 'pip install virtualenv' manually."
            )
            sys.exit(1)


def main():
    print(f"🚀 Starting setup using 'virtualenv' on {sys.platform}...")

    # 1. CHECK PYTHON VERSION
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 10):
        print("❌ Error: You need Python 3.10 or higher.")
        sys.exit(1)
    print(f"✅ Python {major}.{minor} detected.")

    # 2. ENSURE VIRTUALENV IS INSTALLED
    install_virtualenv_if_missing()

    # 3. CREATE THE VIRTUAL ENVIRONMENT
    if not os.path.exists(VENV_NAME):
        print(f"📦 Creating virtual environment '{VENV_NAME}'...")
        try:
            # We explicitly call the 'virtualenv' module here
            subprocess.check_call([sys.executable, "-m", "virtualenv", VENV_NAME])
            print("✅ Virtual environment created.")
        except subprocess.CalledProcessError:
            print("❌ Failed to create environment.")
            sys.exit(1)
    else:
        print("ℹ️  Virtual environment already exists. Skipping creation.")

    # 4. LOCATE THE PIP INSIDE THE VENV
    # We must find the pip.exe INSIDE the new folder, not the system one.
    if platform.system() == "Windows":
        pip_exe = os.path.join(VENV_NAME, "Scripts", "pip.exe")
        python_exe = os.path.join(VENV_NAME, "Scripts", "python.exe")
    else:
        pip_exe = os.path.join(VENV_NAME, "bin", "pip")
        python_exe = os.path.join(VENV_NAME, "bin", "python")

    if not os.path.exists(pip_exe):
        print(f"❌ Error: Could not find pip at {pip_exe}. Creation failed.")
        sys.exit(1)

    # 5. INSTALL LIBRARIES WITH GPU SUPPORT
    if os.path.exists(REQUIREMENTS_FILE):
        print(f"⬇️  Installing libraries from {REQUIREMENTS_FILE}...")
        print("   (This involves a large download for PyTorch. Please be patient...)")

        try:
            # We add --extra-index-url to force NVIDIA CUDA version
            subprocess.check_call(
                [
                    pip_exe,
                    "install",
                    "-r",
                    REQUIREMENTS_FILE,
                    "--extra-index-url",
                    TORCH_INDEX_URL,
                ]
            )
            print("✅ All libraries installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Error installing requirements.")
            sys.exit(1)
    else:
        print(f"⚠️  Warning: {REQUIREMENTS_FILE} not found.")

    # 6. SUCCESS MESSAGE
    print("\n" + "=" * 50)
    print("🎉 SETUP COMPLETE!")
    print("To run the code, copy and paste this:")
    if platform.system() == "Windows":
        print(f"    {VENV_NAME}\\Scripts\\python main.py")
    else:
        print(f"    {VENV_NAME}/bin/python main.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
