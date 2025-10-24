"""
Main application module for the image inpainting system.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List
import argparse

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from .inpainting_models import InpaintingModelFactory, create_mask_from_coordinates, preprocess_image
from .config import ConfigManager, InpaintingConfig, setup_logging

logger = logging.getLogger(__name__)


class ImageInpaintingApp:
    """Main application class for image inpainting."""
    
    def __init__(self, config: Optional[InpaintingConfig] = None):
        """
        Initialize the inpainting application.
        
        Args:
            config: Configuration object
        """
        self.config = config or InpaintingConfig()
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the inpainting model based on configuration."""
        try:
            logger.info(f"Loading {self.config.model_type} model...")
            self.model = InpaintingModelFactory.create_model(
                self.config.model_type, 
                self.config.device
            )
            self.model.load_model(self.config.model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def inpaint_image(
        self, 
        image_path: Path, 
        mask_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        prompt: Optional[str] = None,
        mask_coordinates: Optional[List[Tuple[int, int]]] = None,
        **kwargs
    ) -> Image.Image:
        """
        Perform image inpainting.
        
        Args:
            image_path: Path to input image
            mask_path: Path to mask image (optional)
            output_path: Path to save output (optional)
            prompt: Text prompt for guidance
            mask_coordinates: List of (x, y) coordinates for mask creation
            **kwargs: Additional model parameters
            
        Returns:
            Inpainted image
        """
        try:
            # Load and preprocess image
            logger.info(f"Loading image from {image_path}")
            image = Image.open(image_path).convert("RGB")
            image = preprocess_image(image, self.config.max_image_size)
            
            # Create or load mask
            if mask_coordinates:
                logger.info("Creating mask from coordinates")
                mask = create_mask_from_coordinates(
                    image.size, 
                    mask_coordinates, 
                    self.config.default_brush_size
                )
            elif mask_path:
                logger.info(f"Loading mask from {mask_path}")
                mask = Image.open(mask_path).convert("L")
            else:
                raise ValueError("Either mask_path or mask_coordinates must be provided")
            
            # Use configured prompt if none provided
            if prompt is None:
                prompt = self.config.prompt
            
            # Perform inpainting
            logger.info("Starting inpainting process...")
            result = self.model.inpaint(
                image=image,
                mask=mask,
                prompt=prompt,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                **kwargs
            )
            
            # Save result if output path provided
            if output_path:
                logger.info(f"Saving result to {output_path}")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                result.save(output_path, format=self.config.output_format)
            
            logger.info("Inpainting completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Inpainting failed: {e}")
            raise
    
    def visualize_results(
        self, 
        original: Image.Image, 
        mask: Image.Image, 
        result: Image.Image,
        save_path: Optional[Path] = None
    ) -> None:
        """
        Visualize inpainting results.
        
        Args:
            original: Original image
            mask: Inpainting mask
            result: Inpainted result
            save_path: Path to save visualization
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("Inpainting Mask")
        axes[1].axis('off')
        
        # Result
        axes[2].imshow(result)
        axes[2].set_title("Inpainted Result")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def batch_inpaint(
        self, 
        input_dir: Path, 
        output_dir: Path,
        mask_dir: Optional[Path] = None,
        prompt: Optional[str] = None
    ) -> List[Path]:
        """
        Perform batch inpainting on multiple images.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save results
            mask_dir: Directory containing masks (optional)
            prompt: Text prompt for guidance
            
        Returns:
            List of output file paths
        """
        output_paths = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # Get all image files
        image_files = [
            f for f in input_dir.iterdir() 
            if f.suffix.lower() in image_extensions
        ]
        
        logger.info(f"Found {len(image_files)} images for batch processing")
        
        for image_file in image_files:
            try:
                # Determine mask file
                if mask_dir:
                    mask_file = mask_dir / f"{image_file.stem}_mask{image_file.suffix}"
                    if not mask_file.exists():
                        logger.warning(f"Mask not found for {image_file.name}, skipping")
                        continue
                else:
                    mask_file = None
                
                # Determine output file
                output_file = output_dir / f"{image_file.stem}_inpainted.png"
                
                # Perform inpainting
                self.inpaint_image(
                    image_path=image_file,
                    mask_path=mask_file,
                    output_path=output_file,
                    prompt=prompt
                )
                
                output_paths.append(output_file)
                logger.info(f"Processed {image_file.name}")
                
            except Exception as e:
                logger.error(f"Failed to process {image_file.name}: {e}")
                continue
        
        logger.info(f"Batch processing completed. {len(output_paths)} images processed.")
        return output_paths


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Image Inpainting System")
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument("--mask", type=Path, help="Mask image path")
    parser.add_argument("--output", type=Path, help="Output image path")
    parser.add_argument("--prompt", type=str, help="Text prompt for guidance")
    parser.add_argument("--model", type=str, default="stable_diffusion", 
                       choices=["stable_diffusion", "lama", "traditional"],
                       help="Inpainting model type")
    parser.add_argument("--config", type=Path, help="Configuration file path")
    parser.add_argument("--device", type=str, default="auto", 
                       choices=["auto", "cuda", "cpu", "mps"],
                       help="Device to run on")
    parser.add_argument("--visualize", action="store_true", help="Show visualization")
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.load_config()
    
    # Override config with command line arguments
    if args.model:
        config.model_type = args.model
    if args.device:
        config.device = args.device
    
    # Create application
    app = ImageInpaintingApp(config)
    
    # Perform inpainting
    try:
        result = app.inpaint_image(
            image_path=args.image,
            mask_path=args.mask,
            output_path=args.output,
            prompt=args.prompt
        )
        
        if args.visualize:
            original = Image.open(args.image).convert("RGB")
            mask = Image.open(args.mask).convert("L") if args.mask else None
            app.visualize_results(original, mask, result)
        
        print("Inpainting completed successfully!")
        
    except Exception as e:
        logger.error(f"Application failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
