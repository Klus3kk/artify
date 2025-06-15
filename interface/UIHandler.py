"""
Modern Artify UI - Production Ready
Beautiful, fast, and user-friendly interface for style transfer
"""

import os
os.environ['TORCH_DISABLE_CLASSES'] = '1'
os.environ['STREAMLIT_TORCH_COMPAT'] = '1'

import streamlit as st
import asyncio
import time
import io
import base64
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '.')
sys.path.append(str(Path(__file__).parent.parent))

from core.StyleTransferModel import StyleTransferModel
from utilities.StyleRegistry import StyleRegistry
from utilities.Logger import Logger
import logging

# Setup
logger = Logger.setup_logger(log_file="ui.log", log_level=logging.INFO)

# Page config
st.set_page_config(
    page_title="Artify - AI Style Transfer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown("""
<style>
/* Main theme */
.main {
    padding-top: 2rem;
}

/* Headers */
.main-title {
    font-size: 3rem;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.subtitle {
    text-align: center;
    color: #666;
    font-size: 1.2rem;
    margin-bottom: 2rem;
}

/* Cards */
.style-card {
    border: 2px solid #e0e0e0;
    border-radius: 15px;
    padding: 1rem;
    margin: 0.5rem 0;
    transition: all 0.3s ease;
    cursor: pointer;
}

.style-card:hover {
    border-color: #667eea;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    transform: translateY(-2px);
}

.style-card.selected {
    border-color: #667eea;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
}

/* Progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

/* Buttons */
.stButton > button {
    border-radius: 25px;
    border: none;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
}

/* Results container */
.result-container {
    border-radius: 15px;
    padding: 2rem;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
    margin: 2rem 0;
}

/* Status indicators */
.status-success {
    color: #4caf50;
    font-weight: 600;
}

.status-processing {
    color: #ff9800;
    font-weight: 600;
}

.status-error {
    color: #f44336;
    font-weight: 600;
}

/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

class ModernArtifyUI:
    def __init__(self):
        self.registry = StyleRegistry()
        self.model = None
        self.current_storage = "local"
        
        # Initialize session state
        if 'processing' not in st.session_state:
            st.session_state.processing = False
        if 'result_image' not in st.session_state:
            st.session_state.result_image = None
        if 'selected_style' not in st.session_state:
            st.session_state.selected_style = None
        if 'processing_time' not in st.session_state:
            st.session_state.processing_time = 0
    
    def render_header(self):
        """Render the main header"""
        st.markdown('<h1 class="main-title">Artify</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Transform your photos with AI-powered style transfer</p>', unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar with settings"""
        with st.sidebar:
            st.title("⚙️ Settings")
            
            # Storage backend selection
            st.subheader("Model Storage")
            storage_options = ["local", "s3", "huggingface"]
            self.current_storage = st.selectbox(
                "Storage Backend",
                storage_options,
                help="Choose where to load models from"
            )
            
            # Performance mode
            st.subheader("Performance")
            mode = st.radio(
                "Processing Mode",
                ["fast", "quality"],
                help="Fast: ~0.05s, Quality: ~0.3s"
            )
            
            # Initialize model with selected settings
            if st.button("Initialize Model", type="primary"):
                with st.spinner("Initializing model..."):
                    try:
                        self.model = StyleTransferModel(storage_backend=self.current_storage)
                        if mode == "fast":
                            self.model.set_fast_mode()
                        else:
                            self.model.set_quality_mode()
                        st.success("Model initialized!")
                        
                        # Pre-warm cache with popular styles
                        popular_styles = ["impressionism", "cubism", "abstract"]
                        self.model.warm_up_cache(popular_styles)
                        
                    except Exception as e:
                        st.error(f"Failed to initialize: {e}")
            
            # Model info
            if self.model:
                st.subheader("Model Info")
                try:
                    info = self.model.get_model_info("impressionism")
                    if "total_parameters" in info:
                        st.metric("Parameters", f"{info['total_parameters']:,}")
                        st.metric("Device", info['device'])
                        st.metric("Storage", info['storage_backend'])
                except:
                    pass
            
            # Advanced settings
            with st.expander("🔧 Advanced"):
                auto_enhance = st.checkbox("Auto-enhance input", value=True)
                output_quality = st.slider("Output Quality", 70, 100, 95)
                resize_input = st.checkbox("Resize large images", value=True)
                max_size = st.slider("Max dimension (px)", 512, 2048, 1024)
            
            # About
            st.markdown("---")
            st.subheader("About")
            st.markdown("""
            **Artify** uses cutting-edge neural style transfer with:
            - 🚀 Fast inference (~0.05s)
            - 📱 Small models (~8MB each)
            - ☁️ Multiple storage backends
            - 🎨 Professional quality results
            """)
    
    def render_style_selector(self):
        """Render beautiful style selector"""
        st.subheader("Choose Your Style")
        
        # Get available styles
        styles = self.registry.styles
        
        if not styles:
            st.warning("No style categories found. Please add style images to images/style/")
            return
        
        # Create style grid
        cols = st.columns(4)
        
        for i, (style_name, style_paths) in enumerate(styles.items()):
            with cols[i % 4]:
                # Style preview card
                if style_paths:
                    # Load preview image
                    try:
                        preview_img = Image.open(style_paths[0])
                        preview_img.thumbnail((150, 150))
                        
                        # Display style card
                        selected = st.session_state.selected_style == style_name
                        
                        if st.button(
                            f"{style_name.title()}", 
                            key=f"style_{style_name}",
                            type="primary" if selected else "secondary"
                        ):
                            st.session_state.selected_style = style_name
                            st.rerun()
                        
                        st.image(preview_img, caption=f"{len(style_paths)} examples")
                        
                    except Exception as e:
                        st.error(f"Preview failed: {e}")
        
        # Show selected style
        if st.session_state.selected_style:
            st.success(f"Selected style: **{st.session_state.selected_style.title()}**")
    
    def render_image_uploader(self):
        """Render image upload section"""
        st.subheader("Upload Your Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload the image you want to stylize"
        )
        
        if uploaded_file:
            # Load and display image
            content_image = Image.open(uploaded_file).convert('RGB')
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(content_image, caption="Your Image", use_container_width=True)
            
            with col2:
                # Image info
                st.metric("Size", f"{content_image.size[0]} × {content_image.size[1]}")
                st.metric("Format", uploaded_file.type)
                st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
                
                # Pre-processing options
                if st.checkbox("Auto-enhance", value=True):
                    enhancer = ImageEnhance.Sharpness(content_image)
                    content_image = enhancer.enhance(1.2)
                    
                    enhancer = ImageEnhance.Contrast(content_image)
                    content_image = enhancer.enhance(1.1)
            
            return content_image
        
        return None
    
    def render_generation_section(self, content_image):
        """Render style transfer generation"""
        st.subheader("Generate Stylized Image")
        
        if not self.model:
            st.warning("Please initialize the model first in the sidebar")
            return
        
        if not st.session_state.selected_style:
            st.warning("Please select a style first")
            return
        
        # Generation button
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("Apply Style Transfer", type="primary", use_container_width=True):
                if not st.session_state.processing:
                    self.generate_styled_image(content_image)
    
    def generate_styled_image(self, content_image):
        """Generate styled image with progress tracking"""
        st.session_state.processing = True
        
        # Progress container
        progress_container = st.container()
        
        with progress_container:
            st.markdown('<div class="result-container">', unsafe_allow_html=True)
            st.markdown("### Processing Your Image")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            time_text = st.empty()
            
            try:
                start_time = time.time()
                
                # Step 1: Preparing
                status_text.markdown('<p class="status-processing">📋 Preparing image...</p>', unsafe_allow_html=True)
                progress_bar.progress(20)
                time.sleep(0.5)
                
                # Step 2: Loading model
                status_text.markdown('<p class="status-processing">🧠 Loading model...</p>', unsafe_allow_html=True)
                progress_bar.progress(40)
                time.sleep(0.5)
                
                # Step 3: Style transfer
                status_text.markdown('<p class="status-processing">🎨 Applying style transfer...</p>', unsafe_allow_html=True)
                progress_bar.progress(60)
                
                # Actual style transfer
                styled_image = self.model.apply_style(
                    content_image, 
                    style_category=st.session_state.selected_style
                )
                
                progress_bar.progress(90)
                
                # Step 4: Finalizing
                status_text.markdown('<p class="status-processing">✨ Finalizing...</p>', unsafe_allow_html=True)
                progress_bar.progress(100)
                
                processing_time = time.time() - start_time
                st.session_state.processing_time = processing_time
                st.session_state.result_image = styled_image
                
                # Success
                status_text.markdown('<p class="status-success">✅ Style transfer complete!</p>', unsafe_allow_html=True)
                time_text.markdown(f"**Processing time:** {processing_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Style transfer failed: {e}")
                status_text.markdown(f'<p class="status-error">❌ Error: {e}</p>', unsafe_allow_html=True)
                progress_bar.progress(0)
            
            finally:
                st.session_state.processing = False
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Auto-refresh to show results
                time.sleep(1)
                st.rerun()
    
    def render_results(self, content_image):
        """Render results section"""
        if st.session_state.result_image:
            st.subheader("Your Stylized Image")
            
            # Results container
            st.markdown('<div class="result-container">', unsafe_allow_html=True)
            
            # Before/After comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original**")
                st.image(content_image, use_container_width=True)
            
            with col2:
                st.markdown("**Stylized**")
                st.image(st.session_state.result_image, use_container_width=True)
            
            # Stats
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Style", st.session_state.selected_style.title())
            with col2:
                st.metric("Processing Time", f"{st.session_state.processing_time:.2f}s")
            with col3:
                storage_info = self.current_storage.title()
                st.metric("Model Source", storage_info)
            
            # Download section
            st.subheader("Download Options")
            
            col1, col2, col3 = st.columns(3)
            
            # High quality download
            with col1:
                img_bytes = io.BytesIO()
                st.session_state.result_image.save(img_bytes, format='PNG', quality=100)
                st.download_button(
                    "Download PNG (High Quality)",
                    data=img_bytes.getvalue(),
                    file_name=f"artify_{st.session_state.selected_style}_hq.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            # Standard quality download
            with col2:
                img_bytes = io.BytesIO()
                st.session_state.result_image.save(img_bytes, format='JPEG', quality=95)
                st.download_button(
                    "Download JPEG (Standard)",
                    data=img_bytes.getvalue(),
                    file_name=f"artify_{st.session_state.selected_style}.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
            
            # Social media size
            with col3:
                social_img = st.session_state.result_image.copy()
                social_img.thumbnail((1080, 1080), Image.Resampling.LANCZOS)
                img_bytes = io.BytesIO()
                social_img.save(img_bytes, format='JPEG', quality=90)
                st.download_button(
                    "Download Social (1080px)",
                    data=img_bytes.getvalue(),
                    file_name=f"artify_{st.session_state.selected_style}_social.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
            
            # Clear results button
            if st.button("Start Over", use_container_width=True):
                st.session_state.result_image = None
                st.session_state.selected_style = None
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def render_gallery(self):
        """Render example gallery"""
        with st.expander("Example Gallery", expanded=False):
            st.markdown("See what Artify can do with different styles:")
            
            # Create example grid (you can add real examples)
            examples = [
                {"style": "Impressionism", "desc": "Soft, dreamy brushstrokes"},
                {"style": "Cubism", "desc": "Geometric, abstract forms"},
                {"style": "Abstract", "desc": "Bold colors and shapes"},
                {"style": "Expressionism", "desc": "Emotional, vibrant colors"}
            ]
            
            cols = st.columns(len(examples))
            for i, example in enumerate(examples):
                with cols[i]:
                    st.markdown(f"**{example['style']}**")
                    st.markdown(f"*{example['desc']}*")
    
    def run(self):
        """Main app runner"""
        self.render_header()
        self.render_sidebar()
        
        # Main content
        content_image = self.render_image_uploader()
        
        if content_image:
            self.render_style_selector()
            
            if st.session_state.selected_style:
                self.render_generation_section(content_image)
                
                if st.session_state.result_image:
                    self.render_results(content_image)
        
        self.render_gallery()
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #666;'>"
            "Made with ❤️ using Artify | "
            "<a href='#' style='color: #667eea;'>Documentation</a> | "
            "<a href='#' style='color: #667eea;'>GitHub</a>"
            "</div>",
            unsafe_allow_html=True
        )

# Run the app
if __name__ == "__main__":
    ui = ModernArtifyUI()
    ui.run()