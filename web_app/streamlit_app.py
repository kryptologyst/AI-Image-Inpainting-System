"""
Streamlit web interface for the image inpainting system.
"""

import streamlit as st
import logging
from pathlib import Path
from typing import Optional, List, Tuple
import tempfile
import io

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.inpainting_models import InpaintingModelFactory, create_mask_from_coordinates
from src.config import ConfigManager, InpaintingConfig, get_model_info, setup_logging

# Configure logging
setup_logging("INFO")
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Image Inpainting System",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .model-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f9f9f9;
    }
    .stButton > button {
        width: 100%;
        background-color: #667eea;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #5a6fd8;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_type: str, device: str = "auto"):
    """Load and cache the inpainting model."""
    try:
        model = InpaintingModelFactory.create_model(model_type, device)
        
        # Get model name based on type
        model_info = get_model_info()
        model_name = model_info.get(model_type, {}).get("model_id", "runwayml/stable-diffusion-inpainting")
        
        model.load_model(model_name)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def create_interactive_mask(image: Image.Image, brush_size: int = 20) -> Image.Image:
    """Create an interactive mask using Streamlit's drawing capabilities."""
    st.subheader("🎨 Create Inpainting Mask")
    
    # Convert image to numpy array for drawing
    img_array = np.array(image)
    
    # Create a drawing canvas
    mask_data = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",  # White for mask areas
        stroke_width=brush_size,
        stroke_color="rgba(255, 255, 255, 1)",
        background_image=image,
        update_streamlit=True,
        height=image.height,
        width=image.width,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    if mask_data.image_data is not None:
        # Convert the drawing to a mask
        mask_array = mask_data.image_data[:, :, 3]  # Alpha channel
        mask_array = (mask_array > 0).astype(np.uint8) * 255
        mask = Image.fromarray(mask_array, mode="L")
        return mask
    
    return None


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🎨 AI Image Inpainting System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the AI Image Inpainting System! This application uses state-of-the-art 
    machine learning models to intelligently fill in missing or damaged parts of images.
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection
    model_info = get_model_info()
    model_options = list(model_info.keys())
    
    selected_model = st.sidebar.selectbox(
        "🤖 Select Inpainting Model",
        model_options,
        index=0,
        help="Choose the AI model for inpainting"
    )
    
    # Display model information
    if selected_model in model_info:
        info = model_info[selected_model]
        with st.sidebar.expander("ℹ️ Model Information"):
            st.write(f"**Name:** {info['name']}")
            st.write(f"**Description:** {info['description']}")
            st.write(f"**Best for:** {info['best_for']}")
    
    # Device selection
    device_options = ["auto", "cpu"]
    if st.sidebar.checkbox("Enable GPU (if available)", value=True):
        device_options.extend(["cuda", "mps"])
    
    device = st.sidebar.selectbox("🖥️ Device", device_options, index=0)
    
    # Advanced settings
    with st.sidebar.expander("🔧 Advanced Settings"):
        prompt = st.text_area(
            "📝 Prompt (optional)",
            value="restore the missing part naturally",
            help="Text description to guide the inpainting process"
        )
        
        num_steps = st.slider(
            "🔄 Inference Steps",
            min_value=5,
            max_value=50,
            value=20,
            help="More steps = better quality but slower"
        )
        
        guidance_scale = st.slider(
            "🎯 Guidance Scale",
            min_value=1.0,
            max_value=20.0,
            value=7.5,
            step=0.5,
            help="How closely to follow the prompt"
        )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload an image that needs inpainting"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Mask creation options
            st.subheader("🎭 Mask Creation")
            
            mask_option = st.radio(
                "Choose mask creation method:",
                ["Upload mask image", "Draw mask interactively"],
                help="Select how you want to create the inpainting mask"
            )
            
            mask = None
            
            if mask_option == "Upload mask image":
                mask_file = st.file_uploader(
                    "Upload mask image",
                    type=['png', 'jpg', 'jpeg'],
                    help="White areas will be inpainted, black areas will be preserved"
                )
                
                if mask_file is not None:
                    mask = Image.open(mask_file).convert("L")
                    st.image(mask, caption="Uploaded Mask", use_column_width=True)
            
            else:  # Interactive drawing
                st.info("Draw on the image below to create your mask. White areas will be inpainted.")
                
                # Resize image for better drawing experience
                max_size = 512
                if max(image.size) > max_size:
                    ratio = max_size / max(image.size)
                    new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                    draw_image = image.resize(new_size, Image.Resampling.LANCZOS)
                else:
                    draw_image = image
                
                # Create drawing interface
                brush_size = st.slider("🖌️ Brush Size", 5, 50, 20)
                
                # Simple drawing interface using coordinates
                st.write("Click on the image to add mask points:")
                
                # Create a clickable image
                img_array = np.array(draw_image)
                
                # Use streamlit's experimental features for drawing
                if st.button("🎨 Start Drawing Mode"):
                    st.session_state.drawing_mode = True
                    st.session_state.mask_points = []
                
                if st.session_state.get('drawing_mode', False):
                    st.write("Drawing mode active - click on the image to add mask points")
                    
                    # Simple coordinate input for now
                    col_x, col_y = st.columns(2)
                    with col_x:
                        x = st.number_input("X coordinate", 0, draw_image.width-1, 0)
                    with col_y:
                        y = st.number_input("Y coordinate", 0, draw_image.height-1, 0)
                    
                    if st.button("Add Point"):
                        if 'mask_points' not in st.session_state:
                            st.session_state.mask_points = []
                        st.session_state.mask_points.append((x, y))
                        st.success(f"Added point at ({x}, {y})")
                    
                    if st.session_state.mask_points:
                        st.write(f"Mask points: {st.session_state.mask_points}")
                        
                        if st.button("Create Mask"):
                            mask = create_mask_from_coordinates(
                                draw_image.size,
                                st.session_state.mask_points,
                                brush_size
                            )
                            st.image(mask, caption="Generated Mask", use_column_width=True)
    
    with col2:
        st.header("🎯 Inpainting Results")
        
        if uploaded_file is not None and mask is not None:
            # Process button
            if st.button("🚀 Start Inpainting", type="primary"):
                with st.spinner("Loading model and processing image..."):
                    try:
                        # Load model
                        model = load_model(selected_model, device)
                        
                        if model is None:
                            st.error("Failed to load model. Please try again.")
                            return
                        
                        # Resize mask to match original image if needed
                        if mask.size != image.size:
                            mask = mask.resize(image.size, Image.Resampling.LANCZOS)
                        
                        # Perform inpainting
                        with st.spinner("Performing inpainting..."):
                            result = model.inpaint(
                                image=image,
                                mask=mask,
                                prompt=prompt,
                                num_inference_steps=num_steps,
                                guidance_scale=guidance_scale
                            )
                        
                        # Display results
                        st.success("✅ Inpainting completed!")
                        
                        # Show comparison
                        col_orig, col_mask, col_result = st.columns(3)
                        
                        with col_orig:
                            st.image(image, caption="Original", use_column_width=True)
                        
                        with col_mask:
                            st.image(mask, caption="Mask", use_column_width=True)
                        
                        with col_result:
                            st.image(result, caption="Result", use_column_width=True)
                        
                        # Download button
                        buf = io.BytesIO()
                        result.save(buf, format="PNG")
                        byte_im = buf.getvalue()
                        
                        st.download_button(
                            label="📥 Download Result",
                            data=byte_im,
                            file_name="inpainted_image.png",
                            mime="image/png"
                        )
                        
                    except Exception as e:
                        st.error(f"Inpainting failed: {e}")
                        logger.error(f"Inpainting error: {e}")
        
        else:
            st.info("👆 Please upload an image and create a mask to start inpainting")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 🚀 Features
    - **Multiple AI Models**: Choose from Stable Diffusion, LaMa, and traditional methods
    - **Interactive Masking**: Draw masks directly on your images
    - **High Quality**: State-of-the-art generative AI for realistic results
    - **Easy to Use**: Simple web interface for everyone
    
    ### 📚 How to Use
    1. Upload an image that needs inpainting
    2. Create a mask (white = inpaint, black = keep)
    3. Choose your preferred AI model
    4. Click "Start Inpainting" and wait for results
    5. Download your inpainted image
    """)


# Import streamlit canvas for drawing (if available)
try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st_canvas = None
    st.warning("For interactive drawing, install: pip install streamlit-drawable-canvas")


if __name__ == "__main__":
    main()
