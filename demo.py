#!/usr/bin/env python3
"""
Demo script for the Image Inpainting System

This script demonstrates the capabilities of the inpainting system
using synthetic data and traditional inpainting methods.
"""

import logging
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_generator import generate_sample_dataset
from src.inpainting_models import TraditionalInpainting
from src.config import setup_logging
from PIL import Image

# Setup logging
setup_logging("INFO")
logger = logging.getLogger(__name__)


def main():
    """Run the demo."""
    print("🎨 Image Inpainting System Demo")
    print("=" * 40)
    
    # Create data directory
    data_dir = Path("data/synthetic_dataset")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📊 Generating synthetic dataset...")
    try:
        generate_sample_dataset(data_dir, num_images_per_type=2)
        print("✅ Dataset generated successfully!")
    except Exception as e:
        print(f"❌ Failed to generate dataset: {e}")
        return 1
    
    # Find generated images
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"
    
    if not images_dir.exists() or not masks_dir.exists():
        print("❌ Generated data not found")
        return 1
    
    image_files = list(images_dir.glob("*.png"))
    if not image_files:
        print("❌ No images found")
        return 1
    
    print(f"\n🖼️  Found {len(image_files)} images")
    
    # Test traditional inpainting
    print("\n🔧 Testing traditional inpainting...")
    try:
        model = TraditionalInpainting("cpu")
        model.load_model()
        print("✅ Model loaded successfully")
        
        # Process first image
        image_file = image_files[0]
        mask_file = masks_dir / f"{image_file.stem}_mask.png"
        
        if not mask_file.exists():
            print(f"❌ Mask not found for {image_file.name}")
            return 1
        
        print(f"📸 Processing {image_file.name}...")
        
        # Load images
        image = Image.open(image_file).convert("RGB")
        mask = Image.open(mask_file).convert("L")
        
        # Perform inpainting
        result = model.inpaint(image, mask)
        
        # Save result
        results_dir = data_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        result_path = results_dir / f"{image_file.stem}_result.png"
        result.save(result_path)
        
        print(f"✅ Inpainting completed! Result saved to {result_path}")
        
        # Display image info
        print(f"\n📊 Image Information:")
        print(f"   Original: {image.size} pixels")
        print(f"   Mask: {mask.size} pixels")
        print(f"   Result: {result.size} pixels")
        
    except Exception as e:
        print(f"❌ Inpainting failed: {e}")
        logger.error(f"Inpainting error: {e}")
        return 1
    
    print("\n🎉 Demo completed successfully!")
    print("\n📁 Generated files:")
    print(f"   Images: {images_dir}")
    print(f"   Masks: {masks_dir}")
    print(f"   Results: {data_dir / 'results'}")
    
    print("\n🚀 Next steps:")
    print("   1. Run 'streamlit run web_app/streamlit_app.py' for web interface")
    print("   2. Run 'python src/app.py --help' for CLI options")
    print("   3. Check the README.md for more information")
    
    return 0


if __name__ == "__main__":
    exit(main())
