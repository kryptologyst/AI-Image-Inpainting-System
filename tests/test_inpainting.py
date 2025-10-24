"""
Test suite for the image inpainting system.
"""

import pytest
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

from src.inpainting_models import (
    InpaintingModelFactory,
    create_mask_from_coordinates,
    preprocess_image,
    TraditionalInpainting
)
from src.config import InpaintingConfig, ConfigManager
from src.data_generator import SyntheticDataGenerator


class TestInpaintingModels:
    """Test cases for inpainting models."""
    
    def test_model_factory(self):
        """Test model factory creation."""
        # Test traditional model (doesn't require GPU)
        model = InpaintingModelFactory.create_model("traditional", "cpu")
        assert model is not None
        assert model.device == "cpu"
    
    def test_traditional_inpainting(self):
        """Test traditional inpainting functionality."""
        model = TraditionalInpainting("cpu")
        model.load_model()
        
        # Create test image and mask
        image = Image.new("RGB", (100, 100), color=(255, 0, 0))
        mask = Image.new("L", (100, 100), color=0)
        
        # Add white area to mask
        mask_draw = Image.new("L", (100, 100), 255)
        mask.paste(mask_draw, (25, 25, 75, 75))
        
        # Perform inpainting
        result = model.inpaint(image, mask)
        
        assert result is not None
        assert result.size == image.size
        assert result.mode == "RGB"
    
    def test_mask_creation(self):
        """Test mask creation from coordinates."""
        size = (200, 200)
        coordinates = [(50, 50), (100, 100), (150, 150)]
        brush_size = 20
        
        mask = create_mask_from_coordinates(size, coordinates, brush_size)
        
        assert mask is not None
        assert mask.size == size
        assert mask.mode == "L"
        
        # Check that mask has white areas
        mask_array = np.array(mask)
        assert np.any(mask_array > 0)
    
    def test_image_preprocessing(self):
        """Test image preprocessing."""
        # Create large image
        large_image = Image.new("RGB", (2000, 2000), color=(255, 0, 0))
        
        # Preprocess
        processed = preprocess_image(large_image, max_size=1024)
        
        assert processed is not None
        assert max(processed.size) <= 1024
        
        # Test with small image (should not change)
        small_image = Image.new("RGB", (100, 100), color=(0, 255, 0))
        processed_small = preprocess_image(small_image, max_size=1024)
        
        assert processed_small.size == small_image.size


class TestConfiguration:
    """Test cases for configuration management."""
    
    def test_config_creation(self):
        """Test configuration object creation."""
        config = InpaintingConfig()
        
        assert config.model_type == "stable_diffusion"
        assert config.device == "auto"
        assert config.num_inference_steps == 20
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        config = InpaintingConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "model_type" in config_dict
        assert "device" in config_dict
        
        # Test deserialization
        new_config = InpaintingConfig.from_dict(config_dict)
        assert new_config.model_type == config.model_type
        assert new_config.device == config.device
    
    def test_config_manager(self):
        """Test configuration manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            
            # Create test config
            config = InpaintingConfig()
            config.model_type = "traditional"
            
            # Save config
            manager = ConfigManager(config_path)
            manager.save_config(config)
            
            # Load config
            loaded_config = manager.load_config()
            
            assert loaded_config.model_type == "traditional"


class TestDataGenerator:
    """Test cases for synthetic data generation."""
    
    def test_data_generator_init(self):
        """Test data generator initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = SyntheticDataGenerator(Path(temp_dir))
            
            assert generator.output_dir.exists()
            assert (generator.output_dir / "images").exists()
            assert (generator.output_dir / "masks").exists()
            assert (generator.output_dir / "results").exists()
    
    def test_gradient_background(self):
        """Test gradient background creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = SyntheticDataGenerator(Path(temp_dir))
            
            bg = generator._create_gradient_background((100, 100))
            
            assert bg is not None
            assert bg.size == (100, 100)
            assert bg.mode == "RGB"
    
    def test_simple_shapes_generation(self):
        """Test simple shapes generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = SyntheticDataGenerator(Path(temp_dir))
            
            image_paths = generator.generate_simple_shapes(num_images=2)
            
            assert len(image_paths) == 2
            for path in image_paths:
                assert path.exists()
                assert path.suffix == ".png"
    
    def test_mask_creation(self):
        """Test mask creation for images."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = SyntheticDataGenerator(Path(temp_dir))
            
            # Generate test images
            image_paths = generator.generate_simple_shapes(num_images=2)
            
            # Create masks
            mask_paths = generator.create_damage_masks(image_paths)
            
            assert len(mask_paths) == 2
            for path in mask_paths:
                assert path.exists()
                assert path.suffix == ".png"
                
                # Check mask content
                mask = Image.open(path)
                assert mask.mode == "L"
                
                # Check that mask has some white areas
                mask_array = np.array(mask)
                assert np.any(mask_array > 0)


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_traditional(self):
        """Test end-to-end traditional inpainting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Generate test data
            generator = SyntheticDataGenerator(temp_path)
            image_paths = generator.generate_simple_shapes(num_images=1)
            mask_paths = generator.create_damage_masks(image_paths)
            
            # Create model
            model = TraditionalInpainting("cpu")
            model.load_model()
            
            # Load images
            image = Image.open(image_paths[0]).convert("RGB")
            mask = Image.open(mask_paths[0]).convert("L")
            
            # Perform inpainting
            result = model.inpaint(image, mask)
            
            # Save result
            result_path = temp_path / "result.png"
            result.save(result_path)
            
            assert result_path.exists()
            assert result.size == image.size


if __name__ == "__main__":
    pytest.main([__file__])
