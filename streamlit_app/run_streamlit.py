#!/usr/bin/env python3
"""
Streamlit App Launcher for Fish Classification
This script sets up and runs the Streamlit application
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['streamlit', 'tensorflow', 'plotly', 'pillow']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_missing_packages(packages):
    """Install missing packages"""
    if packages:
        print(f"Installing missing packages: {', '.join(packages)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)

def setup_streamlit_config():
    """Create Streamlit configuration file"""
    config_dir = Path.home() / ".streamlit"
    config_dir.mkdir(exist_ok=True)
    
    config_content = """
[server]
port = 8501
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#4CAF50"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
"""
    
    config_file = config_dir / "config.toml"
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Streamlit config created: {config_file}")

def main():
    """Main launcher function"""
    print("🚀 Fish Classification Streamlit App Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    expected_files = ['app.py']  # The main Streamlit app file
    
    # Check requirements
    print("📦 Checking requirements...")
    missing = check_requirements()
    if missing:
        install_missing_packages(missing)
    
    print("✅ All packages available")
    
    # Setup Streamlit config
    setup_streamlit_config()
    
    # Check if models exist
    base_dir = current_dir.parent if current_dir.name != 'fish_classification_project' else current_dir
    models_dir = base_dir / "results" / "models"
    
    if not models_dir.exists() or not list(models_dir.glob("*.h5")):
        print("⚠️  Warning: No trained models found!")
        print(f"   Expected location: {models_dir}")
        print("   Please run the training pipeline first.")
    else:
        model_files = list(models_dir.glob("*.h5"))
        print(f"✅ Found {len(model_files)} trained models")
    
    # Launch Streamlit
    print("\n🌐 Launching Streamlit app...")
    print("   • URL: http://localhost:8501")
    print("   • Press Ctrl+C to stop")
    print("\n" + "=" * 50)
    
    try:
        # Run Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")

if __name__ == "__main__":
    main()