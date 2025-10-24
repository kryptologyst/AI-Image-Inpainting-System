# AI Image Inpainting System

A comprehensive image inpainting system that uses state-of-the-art AI models to intelligently fill in missing or damaged parts of images. This project supports multiple inpainting approaches including Stable Diffusion, LaMa-style models, and traditional OpenCV methods.

## Features

- **Multiple AI Models**: Choose from Stable Diffusion Inpainting, LaMa-style models, and traditional OpenCV methods
- **High Quality Results**: State-of-the-art generative AI for realistic and natural-looking inpainting
- **Web Interface**: Easy-to-use Streamlit web application for interactive inpainting
- **CLI Support**: Command-line interface for batch processing and automation
- **Configurable**: YAML/JSON configuration files for easy customization
- **Visualization**: Built-in visualization tools to compare original, mask, and results
- **Synthetic Data**: Tools to generate test datasets for development and testing
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/AI-Image-Inpainting-System.git
   cd AI-Image-Inpainting-System
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate sample data** (optional):
   ```bash
   python src/data_generator.py
   ```

### Web Interface (Recommended)

Launch the Streamlit web application:

```bash
streamlit run web_app/streamlit_app.py
```

Then open your browser to `http://localhost:8501` and start inpainting!

### Command Line Interface

```bash
# Basic usage
python src/app.py --image path/to/image.png --mask path/to/mask.png --output result.png

# With custom model and settings
python src/app.py \
    --image data/synthetic_dataset/images/shapes_001.png \
    --mask data/synthetic_dataset/masks/shapes_001_mask.png \
    --output results/inpainted.png \
    --model stable_diffusion \
    --prompt "restore the missing part naturally" \
    --device cuda
```

## 📁 Project Structure

```
0229_Image_inpainting_system/
├── src/                          # Source code
│   ├── inpainting_models.py     # Core inpainting models
│   ├── config.py                # Configuration management
│   ├── app.py                   # Main application and CLI
│   └── data_generator.py        # Synthetic data generation
├── web_app/                     # Web interface
│   └── streamlit_app.py        # Streamlit application
├── data/                        # Data directory
│   └── synthetic_dataset/      # Generated test data
├── config/                      # Configuration files
│   └── default_config.yaml     # Default settings
├── tests/                       # Test files
├── models/                      # Model cache directory
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## Supported Models

### 1. Stable Diffusion Inpainting
- **Best for**: General purpose, creative inpainting
- **Model**: `runwayml/stable-diffusion-inpainting`
- **Features**: Text-guided inpainting, high quality results

### 2. Stable Diffusion XL Inpainting
- **Best for**: High-resolution images, detailed inpainting
- **Model**: `stabilityai/stable-diffusion-xl-base-1.0`
- **Features**: Higher resolution support, better detail preservation

### 3. LaMa-style Inpainting
- **Best for**: Large missing regions, architectural images
- **Model**: `kandinsky-community/kandinsky-2-2-decoder-inpaint`
- **Features**: Excellent for large masks, fast inference

### 4. Traditional OpenCV Methods
- **Best for**: Small repairs, real-time applications
- **Methods**: Telea, Navier-Stokes
- **Features**: Fast, no GPU required, good for simple cases

## Usage Examples

### Web Interface

1. **Upload Image**: Select an image that needs inpainting
2. **Create Mask**: Either upload a mask image or draw one interactively
3. **Choose Model**: Select your preferred AI model
4. **Configure Settings**: Adjust parameters like inference steps and guidance scale
5. **Start Inpainting**: Click the button and wait for results
6. **Download Result**: Save your inpainted image

### Programmatic Usage

```python
from src.inpainting_models import InpaintingModelFactory
from src.config import ConfigManager
from PIL import Image

# Load configuration
config_manager = ConfigManager()
config = config_manager.load_config()

# Create model
model = InpaintingModelFactory.create_model("stable_diffusion", "cuda")
model.load_model()

# Load images
image = Image.open("input.png").convert("RGB")
mask = Image.open("mask.png").convert("L")

# Perform inpainting
result = model.inpaint(
    image=image,
    mask=mask,
    prompt="restore the missing part naturally",
    num_inference_steps=20,
    guidance_scale=7.5
)

# Save result
result.save("output.png")
```

### Batch Processing

```python
from src.app import ImageInpaintingApp
from pathlib import Path

app = ImageInpaintingApp()

# Process multiple images
output_paths = app.batch_inpaint(
    input_dir=Path("data/input_images"),
    output_dir=Path("data/results"),
    mask_dir=Path("data/masks"),
    prompt="restore naturally"
)
```

## Configuration

The system uses YAML configuration files for easy customization. Create a `config/local_config.yaml`:

```yaml
# Model settings
model_type: "stable_diffusion"
model_name: "runwayml/stable-diffusion-inpainting"
device: "auto"

# Inpainting parameters
prompt: "restore the missing part naturally"
num_inference_steps: 20
guidance_scale: 7.5

# Image processing
max_image_size: 1024
output_format: "PNG"

# UI settings
default_brush_size: 20
show_intermediate_results: true
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_inpainting_models.py
```

## Performance Tips

### GPU Optimization
- Use CUDA if available: `--device cuda`
- Consider using `xformers` for faster attention
- Use `torch.compile()` for PyTorch 2.0+ optimization

### Memory Management
- Reduce `max_image_size` for large images
- Use `torch.float16` for GPU inference
- Process images in batches for large datasets

### Quality vs Speed
- Increase `num_inference_steps` for better quality
- Use `guidance_scale` between 7-10 for balanced results
- Traditional methods are fastest but lower quality

## 🔧 Development

### Code Style
- Follow PEP 8 guidelines
- Use type hints throughout
- Add comprehensive docstrings
- Run `black` for code formatting

### Adding New Models

1. Create a new class inheriting from `BaseInpaintingModel`
2. Implement `load_model()` and `inpaint()` methods
3. Add to `InpaintingModelFactory`
4. Update configuration and documentation

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the diffusers library and model hosting
- [Stability AI](https://stability.ai/) for Stable Diffusion models
- [OpenCV](https://opencv.org/) for traditional inpainting methods
- [Streamlit](https://streamlit.io/) for the web interface

## Support

- **Issues**: Report bugs and request features on GitHub Issues
- **Discussions**: Join community discussions on GitHub Discussions
- **Documentation**: Check the inline code documentation for detailed API reference

## Future Enhancements

- [ ] Support for video inpainting
- [ ] Real-time inpainting capabilities
- [ ] Integration with more model architectures
- [ ] Advanced mask creation tools
- [ ] Batch processing optimization
- [ ] Model fine-tuning utilities
- [ ] API server deployment
- [ ] Mobile app interface


# AI-Image-Inpainting-System
# AI-Image-Inpainting-System
