import streamlit as st
from core.StyleTransferModel import StyleTransferModel
from core.ImageProcessor import ImageProcessor
from utilities.StyleRegistry import StyleRegistry
from PIL import Image
import io
import os
import logging
import time
import torch
from utilities.Logger import Logger
import threading
import numpy as np

# Set up logger
logger = Logger.setup_logger(log_file="artify_ui.log", log_level=logging.INFO)

# Global variables for background processing
processing_thread = None
is_processing = False
progress = 0.0
result_image = None

def apply_style_in_background(model, content_image, style_image, style_category):
    """Apply style in a background thread to keep UI responsive"""
    global is_processing, progress, result_image
    
    try:
        # Create an event to simulate progress
        start_time = time.time()
        expected_duration = 30  # seconds, adjust based on your model's performance
        
        def update_progress():
            global progress
            while is_processing and progress < 0.95:
                elapsed = time.time() - start_time
                progress = min(0.95, elapsed / expected_duration)
                time.sleep(0.1)
        
        # Start progress updater thread
        progress_thread = threading.Thread(target=update_progress)
        progress_thread.daemon = True
        progress_thread.start()
        
        # Apply style transfer
        logger.info(f"Starting style transfer with {style_category}")
        result_image = model.apply_style(
            content_image=content_image,
            style_image=style_image,
            iterations=300,  # Reduced iterations for speed
            style_weight=1e6,
            content_weight=1,
            tv_weight=1e-6
        )
        
        # Set progress to 100%
        progress = 1.0
        logger.info("Style transfer completed successfully")
        
    except Exception as e:
        logger.error(f"Error in style transfer: {e}")
    finally:
        is_processing = False
        
def start_processing(model, content_image, style_image, style_category):
    """Start the background processing thread"""
    global processing_thread, is_processing, progress, result_image
    
    # Reset state
    is_processing = True
    progress = 0.0
    result_image = None
    
    # Start processing in a background thread
    processing_thread = threading.Thread(
        target=apply_style_in_background, 
        args=(model, content_image, style_image, style_category)
    )
    processing_thread.daemon = True
    processing_thread.start()

def main():
    # Page configuration
    st.set_page_config(
        page_title="Artify - AI-Powered Style Transfer",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-message {
        padding: 1rem;
        background-color: #E8F5E9;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stProgress > div > div > div {
        background-color: #1E88E5;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("<h1 class='main-header'>Artify: AI-Powered Style Transfer</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Transform your photos into stunning artistic masterpieces!</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("Settings")
        
        # Hugging Face Token
        hf_token = st.text_input("Hugging Face Token", type="password", 
                               help="Required to download models from Hugging Face")
        
        if not hf_token:
            st.warning("Please provide your Hugging Face token to download models.")
        
        # Advanced settings
        st.subheader("Advanced Settings")
        iterations = st.slider("Quality Level", min_value=100, max_value=1000, value=300, step=100,
                              help="Higher values = better quality but slower processing")
        
        show_advanced = st.checkbox("Show Advanced Parameters", value=False)
        
        if show_advanced:
            col1, col2 = st.columns(2)
            with col1:
                style_weight = st.slider("Style Weight", min_value=1, max_value=10, value=5, 
                                       help="Higher values increase style influence")
                
            with col2:
                tv_weight = st.slider("Smoothness", min_value=1, max_value=10, value=5,
                                     help="Higher values create smoother results")
        
        # About section
        st.markdown("---")
        st.subheader("About Artify")
        st.write("""
        Artify uses neural style transfer to apply artistic styles to your images. 
        The algorithm is based on optimizing the output image to match the content 
        of your photo and the style of famous artworks.
        """)
        
        st.markdown("[View Source Code](https://github.com/ClueSec/artify)")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Upload Your Image")
        content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"], 
                                      help="This is the image you want to stylize")
        
        if content_file:
            content_image = Image.open(content_file).convert("RGB")
            st.image(content_image, caption="Your Image", use_column_width=True)
            
            # Image info
            width, height = content_image.size
            st.caption(f"Image dimensions: {width} x {height} pixels")
    
    with col2:
        st.subheader("2. Choose a Style")
        
        # Initialize style registry and get styles
        style_registry = StyleRegistry()
        style_categories = list(style_registry.styles.keys())
        
        style_category = st.selectbox("Select a Style Category", ["Select a Style"] + style_categories)
        
        # Style preview
        if style_category != "Select a Style":
            try:
                style_image_path = style_registry.get_random_style_image(style_category)
                style_image = Image.open(style_image_path).convert("RGB")
                st.image(style_image, caption=f"{style_category.capitalize()} Style", use_column_width=True)
                
                # Show style examples
                st.caption("Click 'Generate' to apply this style to your image.")
            except Exception as e:
                st.error(f"Error loading style: {e}")
    
    # Generate section
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if content_file and style_category != "Select a Style" and hf_token:
            if st.button("Generate Stylized Image", type="primary", use_container_width=True):
                global is_processing, result_image
                
                # Initialize model
                model = StyleTransferModel(hf_token)
                model.ensure_model(style_category)
                
                # Start processing
                start_processing(model, content_image, style_image, style_category)
    
    # Progress and result display
    if is_processing or result_image is not None:
        st.markdown("---")
        
        if is_processing:
            # Show progress bar
            st.subheader("Applying Style Transfer...")
            progress_bar = st.progress(0)
            
            # This will update the progress bar
            progress_placeholder = st.empty()
            
            # Simulate progress updates (Streamlit runs this loop repeatedly)
            progress_bar.progress(progress)
            if progress < 1.0:
                progress_placeholder.text(f"Processing... {int(progress * 100)}%")
                st.experimental_rerun()
            else:
                progress_placeholder.text("Finalizing your image...")
        
        if result_image is not None:
            # Show the result
            st.subheader("Your Stylized Image")
            st.image(result_image, caption="Stylized Result", use_column_width=True)
            
            # Download button
            result_img_io = io.BytesIO()
            result_image.save(result_img_io, format="JPEG", quality=95)
            result_img_io.seek(0)
            
            st.download_button(
                label="Download Stylized Image",
                data=result_img_io,
                file_name=f"artify_{style_category}.jpg",
                mime="image/jpeg",
                use_container_width=True
            )
            
            st.markdown("<div class='success-message'>Style transfer complete! 🎉</div>", unsafe_allow_html=True)
            
            # Comparison view
            if st.checkbox("Show Before/After Comparison"):
                cols = st.columns(2)
                with cols[0]:
                    st.image(content_image, caption="Original", use_column_width=True)
                with cols[1]:
                    st.image(result_image, caption="Stylized", use_column_width=True)
    
    # Tips section
    with st.expander("Tips for Best Results"):
        st.markdown("""
        - **Image Selection**: Choose images with clear subjects and good lighting
        - **Style Matching**: Some styles work better with certain types of images
        - **Resolution**: Higher resolution images produce better details but take longer to process
        - **Quality Level**: Adjust the quality slider for better results at the cost of processing time
        """)

if __name__ == "__main__":
    main()