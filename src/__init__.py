"""
Image Inpainting System

A modern, comprehensive image inpainting system using state-of-the-art AI models.
"""

__version__ = "1.0.0"
__author__ = "AI Projects Team"
__description__ = "AI-powered image inpainting system with multiple model support"

from .inpainting_models import (
    BaseInpaintingModel,
    StableDiffusionInpainting,
    LaMaInpainting,
    TraditionalInpainting,
    InpaintingModelFactory,
    create_mask_from_coordinates,
    preprocess_image
)

from .config import (
    InpaintingConfig,
    ConfigManager,
    setup_logging,
    get_model_info
)

from .app import ImageInpaintingApp

__all__ = [
    "BaseInpaintingModel",
    "StableDiffusionInpainting", 
    "LaMaInpainting",
    "TraditionalInpainting",
    "InpaintingModelFactory",
    "create_mask_from_coordinates",
    "preprocess_image",
    "InpaintingConfig",
    "ConfigManager",
    "setup_logging",
    "get_model_info",
    "ImageInpaintingApp"
]
