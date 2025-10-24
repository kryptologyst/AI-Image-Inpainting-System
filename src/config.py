"""
Configuration and utilities for the image inpainting system.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class InpaintingConfig:
    """Configuration class for inpainting parameters."""
    
    # Model settings
    model_type: str = "stable_diffusion"
    model_name: str = "runwayml/stable-diffusion-inpainting"
    device: str = "auto"
    
    # Inpainting parameters
    prompt: str = "restore the missing part naturally"
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    
    # Image processing
    max_image_size: int = 1024
    output_format: str = "PNG"
    
    # UI settings
    default_brush_size: int = 20
    show_intermediate_results: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'InpaintingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


class ConfigManager:
    """Manages configuration loading and saving."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize config manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or Path("config/default_config.yaml")
        self.config = InpaintingConfig()
    
    def load_config(self, config_path: Optional[Path] = None) -> InpaintingConfig:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded configuration
        """
        if config_path:
            self.config_path = config_path
        
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}. Using defaults.")
            return self.config
        
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.suffix.lower() == '.yaml':
                    config_dict = yaml.safe_load(f)
                elif self.config_path.suffix.lower() == '.json':
                    config_dict = json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {self.config_path.suffix}")
            
            self.config = InpaintingConfig.from_dict(config_dict)
            logger.info(f"Configuration loaded from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}. Using defaults.")
        
        return self.config
    
    def save_config(self, config: InpaintingConfig, config_path: Optional[Path] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save
            config_path: Path to save configuration
        """
        if config_path:
            self.config_path = config_path
        
        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            config_dict = config.to_dict()
            
            with open(self.config_path, 'w') as f:
                if self.config_path.suffix.lower() == '.yaml':
                    yaml.dump(config_dict, f, default_flow_style=False)
                elif self.config_path.suffix.lower() == '.json':
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ValueError(f"Unsupported config format: {self.config_path.suffix}")
            
            logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            raise


def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    logger.info(f"Logging configured with level: {log_level}")


def get_model_info() -> Dict[str, Dict[str, str]]:
    """
    Get information about available models.
    
    Returns:
        Dictionary with model information
    """
    return {
        "stable_diffusion": {
            "name": "Stable Diffusion Inpainting",
            "description": "High-quality generative inpainting with text guidance",
            "model_id": "runwayml/stable-diffusion-inpainting",
            "best_for": "General purpose, creative inpainting"
        },
        "stable_diffusion_xl": {
            "name": "Stable Diffusion XL Inpainting",
            "description": "Higher resolution generative inpainting",
            "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
            "best_for": "High-resolution images, detailed inpainting"
        },
        "lama": {
            "name": "LaMa-style Inpainting",
            "description": "Large mask inpainting with good performance",
            "model_id": "kandinsky-community/kandinsky-2-2-decoder-inpaint",
            "best_for": "Large missing regions, architectural images"
        },
        "traditional": {
            "name": "Traditional OpenCV Inpainting",
            "description": "Fast traditional inpainting methods",
            "model_id": "opencv",
            "best_for": "Small repairs, real-time applications"
        }
    }
