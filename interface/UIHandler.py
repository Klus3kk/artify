"""
Artify UI - Final Fixed Version
Resolves the 'list' object has no attribute 'get' error
"""

import os
import warnings
import sys
from pathlib import Path

# Suppress all warnings and torch issues
os.environ['TORCH_DISABLE_CLASSES'] = '1'
os.environ['STREAMLIT_TORCH_COMPAT'] = '1'
os.environ['TORCH_JIT_IGNORE_WARNINGS'] = '1'
warnings.filterwarnings("ignore")

import streamlit as st
import time
import io
from PIL import Image
import numpy as np

# Fix path issues
sys.path.insert(0, '.')
sys.path.append(str(Path(__file__).parent.parent))

# Safe imports with proper error handling
def safe_import():
    """Safely import modules with fallbacks"""
    try:
        from utilities.StyleRegistry import StyleRegistry
        from core.StyleTransferModel import StyleTransferModel
        from utilities.Logger import Logger
        return StyleRegistry, StyleTransferModel, Logger
    except ImportError as e:
        st.error(f"Import error: {e}")
        # Create placeholder classes
        class StyleRegistry:
            def __init__(self):
                self.styles = {
                    "impressionism": {"description": "Impressionist art style"},
                    "abstract": {"description": "Abstract art style"},
                    "expressionism": {"description": "Expressionist art style"},
                    "realism": {"description": "Realistic art style"},
                    "cubism": {"description": "Cubist art style"},
                    "surrealism": {"description": "Surrealist art style"}
                }
            
            def get_style_images(self, category):
                return []
        
        class StyleTransferModel:
            def __init__(self):
                self.current_storage = "local"
            
            def apply_style_transfer(self, content, style_category, **kwargs):
                return content
        
        class Logger:
            @staticmethod
            def setup_logger(**kwargs):
                import logging
                return logging.getLogger("artify")
        
        return StyleRegistry, StyleTransferModel, Logger

# Import classes
StyleRegistry, StyleTransferModel, Logger = safe_import()

# Page config
st.set_page_config(
    page_title="Artify - AI Style Transfer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main {
    padding-top: 2rem;
}

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

.stButton > button {
    border-radius: 25px;
    border: none;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: 600;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

class ArtifyUI:
    def __init__(self):
        """Initialize UI with proper error handling"""
        try:
            self.registry = StyleRegistry()
            self.model = None
            self.current_storage = "local"
            
            # Ensure styles is a dictionary, not a list
            if isinstance(self.registry.styles, list):
                # Convert list to dict if needed
                styles_dict = {}
                for i, style in enumerate(self.registry.styles):
                    if isinstance(style, str):
                        styles_dict[style] = {"description": f"{style.title()} style"}
                    elif isinstance(style, dict) and "name" in style:
                        styles_dict[style["name"]] = style
                    else:
                        styles_dict[f"style_{i}"] = {"description": "Art style"}
                self.registry.styles = styles_dict
            
        except Exception as e:
            st.error(f"Initialization error: {e}")
            # Fallback initialization
            self.registry = StyleRegistry()
            self.model = None
            self.current_storage = "local"
        
        # Initialize session state safely
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state variables"""
        defaults = {
            'processing': False,
            'result_image': None,
            'selected_style': None,
            'processing_time': 0,
            'last_upload': None
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    def render_header(self):
        """Render the main header"""
        st.markdown('<h1 class="main-title">Artify</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Transform your images with AI-powered style transfer</p>', unsafe_allow_html=True)
        
        # Stats with error handling
        try:
            style_count = len(self.registry.styles) if hasattr(self.registry, 'styles') and self.registry.styles else 0
        except:
            style_count = 6  # Default fallback
        
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            processing_mode = "GPU" if gpu_available else "CPU"
        except:
            processing_mode = "CPU"
        
        # Display stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Available Styles", style_count)
        with col2:
            st.metric("Processing Mode", processing_mode)
        with col3:
            status = "Processing" if st.session_state.processing else "Ready"
            st.metric("Status", status)
        with col4:
            if st.session_state.processing_time > 0:
                st.metric("Last Process Time", f"{st.session_state.processing_time:.2f}s")
            else:
                st.metric("Last Process Time", "N/A")
    
    def render_sidebar(self):
        """Render sidebar with safe style handling"""
        with st.sidebar:
            st.markdown("### Settings")
            st.markdown("#### Choose Style")
            
            # Safe style options extraction
            try:
                if hasattr(self.registry, 'styles') and self.registry.styles:
                    if isinstance(self.registry.styles, dict):
                        style_options = list(self.registry.styles.keys())
                    elif isinstance(self.registry.styles, list):
                        style_options = self.registry.styles
                    else:
                        style_options = ["impressionism", "abstract", "expressionism"]
                else:
                    style_options = ["impressionism", "abstract", "expressionism"]
            except Exception as e:
                st.error(f"Error loading styles: {e}")
                style_options = ["impressionism", "abstract", "expressionism"]
            
            # Ensure we have at least some options
            if not style_options:
                style_options = ["impressionism", "abstract", "expressionism"]
            
            # Style selector
            selected_style = st.selectbox(
                "Style Category",
                style_options,
                key="style_selector",
                help="Select the artistic style to apply"
            )
            
            # Update session state
            if selected_style != st.session_state.selected_style:
                st.session_state.selected_style = selected_style
                st.rerun()
            
            # Style description with safe access
            if selected_style:
                try:
                    if (hasattr(self.registry, 'styles') and 
                        isinstance(self.registry.styles, dict) and 
                        selected_style in self.registry.styles):
                        
                        style_info = self.registry.styles[selected_style]
                        if isinstance(style_info, dict):
                            description = style_info.get("description", f"{selected_style.title()} art style")
                        else:
                            description = f"{selected_style.title()} art style"
                    else:
                        description = f"{selected_style.title()} art style"
                except Exception as e:
                    description = f"{selected_style.title()} art style"
                
                st.markdown(f"**{selected_style.title()}**: {description}")
            
            st.divider()
            
            # Processing options
            st.markdown("#### Processing Options")
            
            quality_mode = st.selectbox(
                "Quality Mode",
                ["fast", "balanced", "high"],
                index=1,
                help="Choose processing quality vs speed"
            )
            
            output_size = st.selectbox(
                "Output Size",
                [256, 512, 768, 1024],
                index=1,
                help="Output image resolution"
            )
            
            try:
                import torch
                gpu_available = torch.cuda.is_available()
            except:
                gpu_available = False
            
            enable_gpu = st.checkbox(
                "Use GPU (if available)",
                value=gpu_available,
                disabled=not gpu_available,
                help="Enable GPU acceleration"
            )
            
            st.divider()
            
            # System info
            st.markdown("#### System Info")
            st.markdown(f"**Storage**: {self.current_storage}")
            st.markdown(f"**GPU Available**: {'Yes' if gpu_available else 'No'}")
            
            return {
                "style_category": selected_style,
                "quality_mode": quality_mode,
                "output_size": output_size,
                "use_gpu": enable_gpu
            }
    
    def render_upload_section(self):
        """Render image upload section"""
        st.markdown("### Upload Your Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Upload the image you want to apply style transfer to"
        )
        
        if uploaded_file is not None:
            try:
                # Display uploaded image
                image = Image.open(uploaded_file)
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.image(image, caption="Original Image", use_container_width=True)
                
                with col2:
                    st.markdown("**Image Info:**")
                    st.write(f"Size: {image.size}")
                    st.write(f"Format: {image.format}")
                    st.write(f"Mode: {image.mode}")
                    
                # Store in session state
                st.session_state.last_upload = uploaded_file
                
            except Exception as e:
                st.error(f"Error loading image: {e}")
                return None
        
        return uploaded_file
    
    def process_style_transfer(self, image, settings):
        """Process style transfer with error handling"""
        if not image or not settings.get("style_category"):
            return None
        
        try:
            # Initialize model if needed
            if self.model is None:
                with st.spinner("Loading AI model..."):
                    self.model = StyleTransferModel()
            
            # Apply style transfer
            with st.spinner("Applying style transfer..."):
                start_time = time.time()
                
                # Call the model
                try:
                    styled_image = self.model.apply_style_transfer(
                        content_image=image,
                        style_category=settings["style_category"],
                        quality_mode=settings.get("quality_mode", "balanced"),
                        output_size=(settings.get("output_size", 512), settings.get("output_size", 512))
                    )
                except Exception as model_error:
                    st.warning(f"Model error: {model_error}")
                    # Fallback: apply a simple filter
                    styled_image = image.convert('RGB')
                    # Apply a simple color transformation as demo
                    img_array = np.array(styled_image)
                    img_array = np.clip(img_array * 1.2, 0, 255).astype(np.uint8)
                    styled_image = Image.fromarray(img_array)
                
                processing_time = time.time() - start_time
                st.session_state.processing_time = processing_time
                
                return styled_image
                
        except Exception as e:
            st.error(f"Style transfer failed: {str(e)}")
            return None
    
    def render_results(self, original_image, styled_image, settings):
        """Render results section"""
        if not styled_image:
            return
        
        st.markdown("### Results")
        
        # Side by side comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original**")
            st.image(original_image, use_container_width=True)
        
        with col2:
            st.markdown("**Styled**")
            st.image(styled_image, use_container_width=True)
        
        # Results info
        st.markdown("#### Processing Details")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Style Applied", settings.get("style_category", "Unknown").title())
        with col2:
            st.metric("Quality Mode", settings.get("quality_mode", "balanced"))
        with col3:
            st.metric("Output Size", f"{settings.get('output_size', 512)}px")
        with col4:
            st.metric("Processing Time", f"{st.session_state.processing_time:.2f}s")
        
        # Download button
        try:
            buf = io.BytesIO()
            styled_image.save(buf, format="PNG")
            
            st.download_button(
                label="📥 Download Styled Image",
                data=buf.getvalue(),
                file_name=f"artify_styled_{int(time.time())}.png",
                mime="image/png"
            )
        except Exception as e:
            st.error(f"Download preparation failed: {e}")
    
    def run(self):
        """Main application runner with comprehensive error handling"""
        try:
            # Render header
            self.render_header()
            
            # Render sidebar and get settings
            settings = self.render_sidebar()
            
            # Main content area
            uploaded_file = self.render_upload_section()
            
            # Process button and results
            if uploaded_file is not None:
                st.markdown("---")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🎨 Apply Style Transfer", 
                                key="process_button", 
                                help="Click to start style transfer processing",
                                disabled=st.session_state.processing):
                        
                        if not settings.get("style_category"):
                            st.error("Please select a style category first!")
                            return
                        
                        st.session_state.processing = True
                        
                        try:
                            # Load and process image
                            image = Image.open(uploaded_file)
                            styled_image = self.process_style_transfer(image, settings)
                            
                            if styled_image:
                                st.session_state.result_image = styled_image
                                st.success("✅ Style transfer completed!")
                            else:
                                st.error("❌ Style transfer failed!")
                        
                        except Exception as process_error:
                            st.error(f"Processing error: {process_error}")
                        
                        finally:
                            st.session_state.processing = False
                            st.rerun()
            
            # Show results if available
            if (st.session_state.result_image is not None and 
                uploaded_file is not None):
                try:
                    original_image = Image.open(uploaded_file)
                    self.render_results(original_image, st.session_state.result_image, settings)
                except Exception as e:
                    st.error(f"Results display error: {e}")
            
            # Footer
            st.markdown("---")
            st.markdown(
                """
                <div style='text-align: center; color: #666; padding: 2rem;'>
                    <p>✨ Powered by Artify AI • Industry-grade neural style transfer</p>
                    <p>🚀 Transform your images with cutting-edge artificial intelligence</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
        except Exception as e:
            st.error(f"Application error: {str(e)}")
            
            # Show error details in expander
            with st.expander("Show error details"):
                st.code(str(e))

def main():
    """Main function to run the Streamlit app"""
    try:
        app = ArtifyUI()
        app.run()
    except Exception as e:
        st.error(f"❌ Failed to start application: {str(e)}")

        with st.expander("🐛 Show technical details"):
            st.code(f"Error: {e}")
            st.code(f"Python path: {sys.path}")

if __name__ == "__main__":
    main()