import streamlit as st
from core.StyleTransferModel import StyleTransferModel
from core.ImageProcessor import ImageProcessor
from PIL import Image

def main():
    st.title("Artify: AI-Powered Image Style Transfer")
    st.write("Upload your content and style images to generate a styled result!")

    content_file = st.file_uploader("Upload Content Image", type=["jpg", "png"])
    style_file = st.file_uploader("Upload Style Image", type=["jpg", "png"])

    if content_file and style_file:
        processor = ImageProcessor()
        model = StyleTransferModel()

        content_image = Image.open(content_file)
        style_image = Image.open(style_file)

        st.image(content_image, caption="Content Image", use_column_width=True)
        st.image(style_image, caption="Style Image", use_column_width=True)

        if st.button("Generate Styled Image"):
            styled_image = model.apply_style(content_image, style_image)
            st.image(styled_image, caption="Styled Image", use_column_width=True)

if __name__ == "__main__":
    main()
