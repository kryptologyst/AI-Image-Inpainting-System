"""
Data generation utilities for creating synthetic datasets and sample images.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional
import random
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generator for synthetic images and masks for testing inpainting models."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize the data generator.
        
        Args:
            output_dir: Directory to save generated data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "masks").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
    
    def generate_simple_shapes(self, num_images: int = 10, size: Tuple[int, int] = (512, 512)) -> List[Path]:
        """
        Generate images with simple geometric shapes.
        
        Args:
            num_images: Number of images to generate
            size: Image dimensions
            
        Returns:
            List of generated image paths
        """
        image_paths = []
        
        for i in range(num_images):
            # Create base image with gradient background
            img = self._create_gradient_background(size)
            draw = ImageDraw.Draw(img)
            
            # Add random shapes
            num_shapes = random.randint(2, 5)
            for _ in range(num_shapes):
                self._add_random_shape(draw, size)
            
            # Add some text
            if random.random() > 0.5:
                self._add_text(draw, size)
            
            # Save image
            img_path = self.output_dir / "images" / f"shapes_{i:03d}.png"
            img.save(img_path)
            image_paths.append(img_path)
            
            logger.info(f"Generated shapes image: {img_path}")
        
        return image_paths
    
    def generate_natural_scenes(self, num_images: int = 10, size: Tuple[int, int] = (512, 512)) -> List[Path]:
        """
        Generate more natural-looking scenes.
        
        Args:
            num_images: Number of images to generate
            size: Image dimensions
            
        Returns:
            List of generated image paths
        """
        image_paths = []
        
        for i in range(num_images):
            # Create sky gradient
            img = self._create_sky_background(size)
            draw = ImageDraw.Draw(img)
            
            # Add landscape elements
            self._add_landscape_elements(draw, size)
            
            # Add some objects
            num_objects = random.randint(1, 3)
            for _ in range(num_objects):
                self._add_natural_object(draw, size)
            
            # Save image
            img_path = self.output_dir / "images" / f"natural_{i:03d}.png"
            img.save(img_path)
            image_paths.append(img_path)
            
            logger.info(f"Generated natural scene: {img_path}")
        
        return image_paths
    
    def generate_architectural_scenes(self, num_images: int = 10, size: Tuple[int, int] = (512, 512)) -> List[Path]:
        """
        Generate architectural scenes with buildings and structures.
        
        Args:
            num_images: Number of images to generate
            size: Image dimensions
            
        Returns:
            List of generated image paths
        """
        image_paths = []
        
        for i in range(num_images):
            # Create building background
            img = self._create_building_background(size)
            draw = ImageDraw.Draw(img)
            
            # Add architectural elements
            self._add_architectural_elements(draw, size)
            
            # Add windows and doors
            self._add_windows_doors(draw, size)
            
            # Save image
            img_path = self.output_dir / "images" / f"architecture_{i:03d}.png"
            img.save(img_path)
            image_paths.append(img_path)
            
            logger.info(f"Generated architectural scene: {img_path}")
        
        return image_paths
    
    def create_damage_masks(self, image_paths: List[Path], damage_types: List[str] = None) -> List[Path]:
        """
        Create damage masks for the generated images.
        
        Args:
            image_paths: List of image paths
            damage_types: Types of damage to simulate
            
        Returns:
            List of generated mask paths
        """
        if damage_types is None:
            damage_types = ["scratch", "hole", "stain", "tear", "burn"]
        
        mask_paths = []
        
        for img_path in image_paths:
            # Load image to get dimensions
            img = Image.open(img_path)
            size = img.size
            
            # Create mask
            mask = Image.new("L", size, 0)
            draw = ImageDraw.Draw(mask)
            
            # Add random damage
            damage_type = random.choice(damage_types)
            self._add_damage_pattern(draw, size, damage_type)
            
            # Save mask
            mask_path = self.output_dir / "masks" / f"{img_path.stem}_mask.png"
            mask.save(mask_path)
            mask_paths.append(mask_path)
            
            logger.info(f"Generated mask: {mask_path}")
        
        return mask_paths
    
    def _create_gradient_background(self, size: Tuple[int, int]) -> Image.Image:
        """Create a gradient background."""
        img = Image.new("RGB", size)
        
        # Create gradient
        for y in range(size[1]):
            # Color transition from top to bottom
            r = int(50 + (200 - 50) * y / size[1])
            g = int(100 + (150 - 100) * y / size[1])
            b = int(150 + (100 - 150) * y / size[1])
            
            for x in range(size[0]):
                img.putpixel((x, y), (r, g, b))
        
        return img
    
    def _create_sky_background(self, size: Tuple[int, int]) -> Image.Image:
        """Create a sky-like background."""
        img = Image.new("RGB", size)
        
        # Sky gradient
        for y in range(size[1]):
            # Sky blue gradient
            intensity = int(135 + (255 - 135) * y / size[1])
            color = (intensity, intensity + 20, intensity + 40)
            
            for x in range(size[0]):
                img.putpixel((x, y), color)
        
        return img
    
    def _create_building_background(self, size: Tuple[int, int]) -> Image.Image:
        """Create a building-like background."""
        img = Image.new("RGB", size)
        
        # Building color
        building_color = (80, 80, 80)
        
        # Fill with building color
        for y in range(size[1]):
            for x in range(size[0]):
                img.putpixel((x, y), building_color)
        
        return img
    
    def _add_random_shape(self, draw: ImageDraw.Draw, size: Tuple[int, int]) -> None:
        """Add a random geometric shape."""
        shape_type = random.choice(["circle", "rectangle", "polygon"])
        
        # Random position and size
        x1 = random.randint(0, size[0] // 2)
        y1 = random.randint(0, size[1] // 2)
        x2 = random.randint(x1 + 50, size[0])
        y2 = random.randint(y1 + 50, size[1])
        
        # Random color
        color = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255)
        )
        
        if shape_type == "circle":
            draw.ellipse([x1, y1, x2, y2], fill=color)
        elif shape_type == "rectangle":
            draw.rectangle([x1, y1, x2, y2], fill=color)
        else:  # polygon
            points = [
                (x1, y1),
                (x2, y1),
                (x2, y2),
                (x1, y2),
                (x1 + (x2-x1)//2, y1 - 20)
            ]
            draw.polygon(points, fill=color)
    
    def _add_text(self, draw: ImageDraw.Draw, size: Tuple[int, int]) -> None:
        """Add random text to the image."""
        try:
            # Try to use a system font
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
        
        text = f"Sample {random.randint(1, 100)}"
        x = random.randint(10, size[0] - 100)
        y = random.randint(10, size[1] - 30)
        
        # Add text with outline
        draw.text((x-1, y-1), text, fill=(0, 0, 0), font=font)
        draw.text((x+1, y+1), text, fill=(0, 0, 0), font=font)
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
    
    def _add_landscape_elements(self, draw: ImageDraw.Draw, size: Tuple[int, int]) -> None:
        """Add landscape elements like trees and mountains."""
        # Add mountains
        mountain_points = []
        for i in range(0, size[0] + 50, 50):
            mountain_points.append((i, size[1] - random.randint(50, 150)))
        
        mountain_points.append((size[0], size[1]))
        mountain_points.append((0, size[1]))
        
        draw.polygon(mountain_points, fill=(100, 80, 60))
        
        # Add trees
        for _ in range(random.randint(3, 8)):
            x = random.randint(0, size[0])
            y = random.randint(size[1] // 2, size[1] - 50)
            
            # Tree trunk
            draw.rectangle([x-5, y, x+5, y+30], fill=(101, 67, 33))
            
            # Tree crown
            draw.ellipse([x-20, y-20, x+20, y+10], fill=(34, 139, 34))
    
    def _add_natural_object(self, draw: ImageDraw.Draw, size: Tuple[int, int]) -> None:
        """Add natural objects like rocks or clouds."""
        obj_type = random.choice(["rock", "cloud"])
        
        if obj_type == "rock":
            x = random.randint(50, size[0] - 50)
            y = random.randint(size[1] // 2, size[1] - 50)
            radius = random.randint(20, 40)
            
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=(120, 120, 120))
        
        else:  # cloud
            x = random.randint(50, size[0] - 100)
            y = random.randint(50, size[1] // 3)
            
            # Multiple circles for cloud shape
            for i in range(3):
                offset_x = random.randint(-20, 20)
                offset_y = random.randint(-10, 10)
                radius = random.randint(15, 25)
                
                draw.ellipse(
                    [x + offset_x - radius, y + offset_y - radius, 
                     x + offset_x + radius, y + offset_y + radius],
                    fill=(200, 200, 200)
                )
    
    def _add_architectural_elements(self, draw: ImageDraw.Draw, size: Tuple[int, int]) -> None:
        """Add architectural elements like walls and roofs."""
        # Add walls
        wall_color = (100, 100, 100)
        
        # Main building
        draw.rectangle([50, size[1]//2, size[0]-50, size[1]-50], fill=wall_color)
        
        # Add roof
        roof_points = [
            (50, size[1]//2),
            (size[0]//2, size[1]//2 - 50),
            (size[0]-50, size[1]//2)
        ]
        draw.polygon(roof_points, fill=(139, 69, 19))
    
    def _add_windows_doors(self, draw: ImageDraw.Draw, size: Tuple[int, int]) -> None:
        """Add windows and doors to buildings."""
        # Add windows
        for i in range(3):
            x = 100 + i * 80
            y = size[1]//2 + 20
            draw.rectangle([x, y, x+30, y+40], fill=(135, 206, 235))
        
        # Add door
        door_x = size[0]//2 - 15
        door_y = size[1] - 100
        draw.rectangle([door_x, door_y, door_x+30, door_y+50], fill=(101, 67, 33))
    
    def _add_damage_pattern(self, draw: ImageDraw.Draw, size: Tuple[int, int], damage_type: str) -> None:
        """Add damage patterns to create masks."""
        if damage_type == "scratch":
            # Horizontal scratch
            y = random.randint(size[1]//4, 3*size[1]//4)
            draw.rectangle([0, y-2, size[0], y+2], fill=255)
        
        elif damage_type == "hole":
            # Circular hole
            x = random.randint(size[0]//4, 3*size[0]//4)
            y = random.randint(size[1]//4, 3*size[1]//4)
            radius = random.randint(20, 50)
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=255)
        
        elif damage_type == "stain":
            # Irregular stain
            center_x = random.randint(size[0]//4, 3*size[0]//4)
            center_y = random.randint(size[1]//4, 3*size[1]//4)
            
            for _ in range(5):
                offset_x = random.randint(-30, 30)
                offset_y = random.randint(-30, 30)
                radius = random.randint(10, 25)
                
                draw.ellipse(
                    [center_x + offset_x - radius, center_y + offset_y - radius,
                     center_x + offset_x + radius, center_y + offset_y + radius],
                    fill=255
                )
        
        elif damage_type == "tear":
            # Diagonal tear
            start_x = random.randint(0, size[0]//2)
            start_y = random.randint(0, size[1]//2)
            end_x = random.randint(size[0]//2, size[0])
            end_y = random.randint(size[1]//2, size[1])
            
            # Create tear shape with multiple lines
            for i in range(10):
                offset = random.randint(-5, 5)
                draw.line([start_x + offset, start_y + offset, end_x + offset, end_y + offset], fill=255, width=3)
        
        else:  # burn
            # Burn mark
            x = random.randint(size[0]//4, 3*size[0]//4)
            y = random.randint(size[1]//4, 3*size[1]//4)
            
            # Multiple overlapping circles
            for i in range(3):
                radius = random.randint(15, 35)
                draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=255)


def generate_sample_dataset(output_dir: Path, num_images_per_type: int = 5) -> None:
    """
    Generate a complete sample dataset for testing.
    
    Args:
        output_dir: Directory to save the dataset
        num_images_per_type: Number of images per type
    """
    generator = SyntheticDataGenerator(output_dir)
    
    logger.info("Generating synthetic dataset...")
    
    # Generate different types of images
    all_image_paths = []
    
    # Simple shapes
    shape_images = generator.generate_simple_shapes(num_images_per_type)
    all_image_paths.extend(shape_images)
    
    # Natural scenes
    natural_images = generator.generate_natural_scenes(num_images_per_type)
    all_image_paths.extend(natural_images)
    
    # Architectural scenes
    arch_images = generator.generate_architectural_scenes(num_images_per_type)
    all_image_paths.extend(arch_images)
    
    # Create masks for all images
    generator.create_damage_masks(all_image_paths)
    
    logger.info(f"Generated {len(all_image_paths)} images with masks")
    
    # Create a dataset info file
    info_file = output_dir / "dataset_info.txt"
    with open(info_file, 'w') as f:
        f.write("Synthetic Image Inpainting Dataset\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total images: {len(all_image_paths)}\n")
        f.write(f"Images per type: {num_images_per_type}\n")
        f.write(f"Image types: shapes, natural, architectural\n")
        f.write(f"Damage types: scratch, hole, stain, tear, burn\n\n")
        f.write("Directory structure:\n")
        f.write("- images/: Original images\n")
        f.write("- masks/: Corresponding masks\n")
        f.write("- results/: Inpainting results (to be generated)\n")


if __name__ == "__main__":
    # Generate sample dataset
    output_dir = Path("data/synthetic_dataset")
    generate_sample_dataset(output_dir, num_images_per_type=3)
