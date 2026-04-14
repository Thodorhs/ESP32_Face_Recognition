import sys
import os
import subprocess
import venv

def check_python_version():
    """Ensure the user is running Python 3.12 or 3.13."""
    major = sys.version_info.major
    minor = sys.version_info.minor
    
    print(f"Detected Python {major}.{minor}")
    
    if major != 3 or minor not in[12, 13]:
        print("\n[ERROR] Unsupported Python version!")
        print("This project requires Python 3.12 or 3.13.")
        print("Please install the correct version and try again.")
        sys.exit(1)
    
    print("[OK] Python version is compatible.")

def create_virtual_environment(venv_dir=".venv"):
    """Create a virtual environment if it doesn't already exist."""
    if not os.path.exists(venv_dir):
        print(f"\nCreating virtual environment in '{venv_dir}'...")
        venv.create(venv_dir, with_pip=True)
        print("[OK] Virtual environment created.")
    else:
        print(f"\n[INFO] Virtual environment '{venv_dir}' already exists. Skipping creation.")
        
    return venv_dir

def install_requirements(venv_dir):
    """Install dependencies from requirements.txt into the virtual environment."""
    req_file = "requirements.txt"
    if not os.path.exists(req_file):
        print(f"\n[ERROR] '{req_file}' not found. Cannot install dependencies.")
        sys.exit(1)

    print(f"\nInstalling requirements from '{req_file}'...")
    
    # Determine the path to the virtual environment's pip executable
    if os.name == 'nt': # Windows
        pip_exe = os.path.join(venv_dir, "Scripts", "pip.exe")
    else:               # Mac/Linux
        pip_exe = os.path.join(venv_dir, "bin", "pip")

    # Run the pip install command inside the venv
    try:
        subprocess.check_call([pip_exe, "install", "-r", req_file])
        print("\n[SUCCESS] Project setup complete!")
    except subprocess.CalledProcessError:
        print("\n[ERROR] Failed to install requirements.")
        sys.exit(1)

def main():
    print("=== ESP32 Face Recognition Setup ===")
    check_python_version()
    venv_dir = create_virtual_environment()
    install_requirements(venv_dir)
    
    print("\nTo activate your environment, run:")
    if os.name == 'nt':
        print(f"    {venv_dir}\\Scripts\\activate")
    else:
        print(f"    source {venv_dir}/bin/activate")

if __name__ == "__main__":
    main()