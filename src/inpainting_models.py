"""
Image Inpainting Models Module

This module provides a unified interface for various state-of-the-art image inpainting models
including Stable Diffusion Inpainting, LaMa, and Paint-by-Example.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, List
from pathlib import Path

import torch
from PIL import Image
import numpy as np
from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionXLInpaintPipeline,
    AutoPipelineForInpainting
)
from transformers import AutoProcessor, AutoModelForImageInpainting
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseInpaintingModel(ABC):
    """Abstract base class for inpainting models."""
    
    def __init__(self, device: str = "auto"):
        """
        Initialize the inpainting model.
        
        Args:
            device: Device to run the model on ("auto", "cuda", "cpu")
        """
        self.device = self._get_device(device)
        self.model = None
        
    def _get_device(self, device: str) -> str:
        """Determine the best available device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    @abstractmethod
    def load_model(self, model_name: str) -> None:
        """Load the inpainting model."""
        pass
    
    @abstractmethod
    def inpaint(
        self, 
        image: Image.Image, 
        mask: Image.Image, 
        prompt: Optional[str] = None,
        **kwargs
    ) -> Image.Image:
        """
        Perform image inpainting.
        
        Args:
            image: Input image
            mask: Inpainting mask (white = inpaint, black = keep)
            prompt: Text prompt for guidance (optional)
            **kwargs: Additional model-specific parameters
            
        Returns:
            Inpainted image
        """
        pass


class StableDiffusionInpainting(BaseInpaintingModel):
    """Stable Diffusion Inpainting model implementation."""
    
    def load_model(self, model_name: str = "runwayml/stable-diffusion-inpainting") -> None:
        """
        Load Stable Diffusion Inpainting model.
        
        Args:
            model_name: Hugging Face model identifier
        """
        try:
            logger.info(f"Loading Stable Diffusion Inpainting model: {model_name}")
            
            # Use appropriate pipeline based on model
            if "xl" in model_name.lower():
                self.model = StableDiffusionXLInpaintPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    use_safetensors=True
                )
            else:
                self.model = StableDiffusionInpaintPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    use_safetensors=True
                )
            
            self.model = self.model.to(self.device)
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def inpaint(
        self, 
        image: Image.Image, 
        mask: Image.Image, 
        prompt: Optional[str] = "restore the missing part naturally",
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        **kwargs
    ) -> Image.Image:
        """
        Perform inpainting using Stable Diffusion.
        
        Args:
            image: Input image
            mask: Inpainting mask
            prompt: Text prompt for guidance
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for prompt adherence
            **kwargs: Additional parameters
            
        Returns:
            Inpainted image
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            logger.info("Starting Stable Diffusion inpainting...")
            
            # Ensure mask is in the correct format
            if mask.mode != "L":
                mask = mask.convert("L")
            
            result = self.model(
                prompt=prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                **kwargs
            ).images[0]
            
            logger.info("Inpainting completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Inpainting failed: {e}")
            raise


class LaMaInpainting(BaseInpaintingModel):
    """LaMa (Large Mask Inpainting) model implementation."""
    
    def load_model(self, model_name: str = "kandinsky-community/kandinsky-2-2-decoder-inpaint") -> None:
        """
        Load LaMa-style inpainting model.
        
        Args:
            model_name: Hugging Face model identifier
        """
        try:
            logger.info(f"Loading LaMa-style model: {model_name}")
            
            # Use AutoPipeline for flexibility
            self.model = AutoPipelineForInpainting.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True
            )
            
            self.model = self.model.to(self.device)
            logger.info("LaMa model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load LaMa model: {e}")
            raise
    
    def inpaint(
        self, 
        image: Image.Image, 
        mask: Image.Image, 
        prompt: Optional[str] = None,
        num_inference_steps: int = 20,
        **kwargs
    ) -> Image.Image:
        """
        Perform inpainting using LaMa-style model.
        
        Args:
            image: Input image
            mask: Inpainting mask
            prompt: Text prompt (optional for LaMa)
            num_inference_steps: Number of denoising steps
            **kwargs: Additional parameters
            
        Returns:
            Inpainted image
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            logger.info("Starting LaMa inpainting...")
            
            # Ensure mask is in the correct format
            if mask.mode != "L":
                mask = mask.convert("L")
            
            result = self.model(
                prompt=prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=num_inference_steps,
                **kwargs
            ).images[0]
            
            logger.info("LaMa inpainting completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"LaMa inpainting failed: {e}")
            raise


class TraditionalInpainting(BaseInpaintingModel):
    """Traditional OpenCV-based inpainting methods."""
    
    def load_model(self, model_name: str = "opencv") -> None:
        """
        Initialize traditional inpainting methods.
        
        Args:
            model_name: Method name (opencv, navier_stokes, etc.)
        """
        self.method_name = model_name
        logger.info(f"Initialized traditional inpainting: {model_name}")
    
    def inpaint(
        self, 
        image: Image.Image, 
        mask: Image.Image, 
        prompt: Optional[str] = None,
        method: str = "telea",
        **kwargs
    ) -> Image.Image:
        """
        Perform traditional inpainting using OpenCV.
        
        Args:
            image: Input image
            mask: Inpainting mask
            prompt: Not used in traditional methods
            method: OpenCV inpainting method ("telea" or "ns")
            **kwargs: Additional parameters
            
        Returns:
            Inpainted image
        """
        try:
            logger.info(f"Starting traditional inpainting with method: {method}")
            
            # Convert PIL images to OpenCV format
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            mask_cv = np.array(mask.convert("L"))
            
            # Choose inpainting method
            if method == "telea":
                inpaint_method = cv2.INPAINT_TELEA
            elif method == "ns":
                inpaint_method = cv2.INPAINT_NS
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Perform inpainting
            result_cv = cv2.inpaint(img_cv, mask_cv, 3, inpaint_method)
            
            # Convert back to PIL
            result = Image.fromarray(cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB))
            
            logger.info("Traditional inpainting completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Traditional inpainting failed: {e}")
            raise


class InpaintingModelFactory:
    """Factory class for creating inpainting models."""
    
    @staticmethod
    def create_model(model_type: str, device: str = "auto") -> BaseInpaintingModel:
        """
        Create an inpainting model instance.
        
        Args:
            model_type: Type of model ("stable_diffusion", "lama", "traditional")
            device: Device to run the model on
            
        Returns:
            Inpainting model instance
        """
        model_type = model_type.lower()
        
        if model_type in ["stable_diffusion", "sd"]:
            return StableDiffusionInpainting(device)
        elif model_type in ["lama", "kandinsky"]:
            return LaMaInpainting(device)
        elif model_type in ["traditional", "opencv"]:
            return TraditionalInpainting(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


def create_mask_from_coordinates(
    image_size: Tuple[int, int], 
    coordinates: List[Tuple[int, int]], 
    brush_size: int = 20
) -> Image.Image:
    """
    Create an inpainting mask from coordinate points.
    
    Args:
        image_size: Size of the image (width, height)
        coordinates: List of (x, y) coordinate tuples
        brush_size: Size of the brush for masking
        
    Returns:
        PIL Image mask
    """
    mask = Image.new("L", image_size, 0)
    
    for x, y in coordinates:
        # Create a circular brush
        brush_mask = Image.new("L", (brush_size * 2, brush_size * 2), 0)
        draw = Image.new("L", (brush_size * 2, brush_size * 2), 255)
        
        # Draw circle
        for i in range(brush_size * 2):
            for j in range(brush_size * 2):
                if (i - brush_size) ** 2 + (j - brush_size) ** 2 <= brush_size ** 2:
                    brush_mask.putpixel((i, j), 255)
        
        # Paste brush onto mask
        mask.paste(brush_mask, (x - brush_size, y - brush_size), brush_mask)
    
    return mask


def preprocess_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """
    Preprocess image for inpainting.
    
    Args:
        image: Input image
        max_size: Maximum dimension size
        
    Returns:
        Preprocessed image
    """
    # Resize if too large
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    return image
