import streamlit as st
from core.StyleTransferModel import StyleTransferModel
from core.ImageProcessor import ImageProcessor
from utilities.StyleRegistry import StyleRegistry
from PIL import Image

def main():
    st.title("Artify: AI-Powered Image Style Transfer")
    st.write("Upload your content image and choose a style category to generate a styled result!")

    style_registry = StyleRegistry()
    style_categories = list(style_registry.styles.keys())
    style_category = st.selectbox("Choose a style category:", style_categories)

    content_file = st.file_uploader("Upload Content Image", type=["jpg", "png"])

    if content_file and style_category:
        processor = ImageProcessor()
        model = StyleTransferModel()

        content_image = Image.open(content_file).convert("RGB")

        style_image_path = style_registry.get_random_style_image(style_category)
        style_image = Image.open(style_image_path).convert("RGB")

        st.image(content_image, caption="Content Image", use_column_width=True)
        st.image(style_image, caption=f"Style Image ({style_category.capitalize()})", use_column_width=True)

        if st.button("Generate Styled Image"):
            st.write("Applying style... Please wait.")
            styled_image = model.apply_style(content_image, style_image)
            st.image(styled_image, caption="Styled Image", use_column_width=True)
            st.success("Style transfer complete! 🎉")

if __name__ == "__main__":
    main()
