#!/usr/bin/env python3
"""
Setup script for the Image Inpainting System

This script helps set up the environment and install dependencies.
"""

import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🚀 Image Inpainting System Setup")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return 1
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("💡 Try running: pip install --upgrade pip")
        return 1
    
    # Create necessary directories
    directories = ["data", "models", "logs", "config"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Generate sample data
    print("\n📊 Generating sample data...")
    if not run_command("python src/data_generator.py", "Generating synthetic dataset"):
        print("⚠️  Sample data generation failed, but you can continue")
    
    # Run tests
    print("\n🧪 Running tests...")
    if not run_command("python -m pytest tests/ -v", "Running test suite"):
        print("⚠️  Some tests failed, but the system should still work")
    
    print("\n🎉 Setup completed!")
    print("\n📚 Next steps:")
    print("   1. Run the demo: python demo.py")
    print("   2. Start web interface: streamlit run web_app/streamlit_app.py")
    print("   3. Try CLI: python src/app.py --help")
    print("   4. Read the README.md for more information")
    
    return 0


if __name__ == "__main__":
    exit(main())
