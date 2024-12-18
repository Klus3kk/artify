import streamlit as st
from core.StyleTransferModel import StyleTransferModel
from core.ImageProcessor import ImageProcessor
from utilities.StyleRegistry import StyleRegistry
from PIL import Image
import io


def main():
    st.sidebar.title("About Artify")
    st.sidebar.write("Artify allows you to apply artistic styles to your images using AI-powered style transfer.")

    st.title("Artify: AI-Powered Image Style Transfer")
    st.write("Upload your content image and choose a style category to generate a styled result!")

    style_registry = StyleRegistry()
    style_categories = list(style_registry.styles.keys())
    style_category = st.selectbox("Choose a style category:", ["Select a Style"] + style_categories)

    content_file = st.file_uploader("Upload Content Image", type=["jpg", "png"])

    if content_file and style_category != "Select a Style":
        try:
            processor = ImageProcessor()
            model = StyleTransferModel()

            content_image = Image.open(content_file).convert("RGB")
            style_image_path = style_registry.get_random_style_image(style_category)
            style_image = Image.open(style_image_path).convert("RGB")

            st.image(content_image, caption="Content Image", use_column_width=True)
            st.image(style_image, caption=f"Style Image ({style_category.capitalize()})", use_column_width=True)

            if st.button("Generate Styled Image"):
                with st.spinner("Applying style... Please wait."):
                    styled_image = model.apply_style(content_image, style_image)
                    st.image(styled_image, caption="Styled Image", use_column_width=True)

                # Convert styled image for download
                styled_image_io = io.BytesIO()
                styled_image.save(styled_image_io, format="JPEG")
                styled_image_io.seek(0)

                st.download_button(
                    label="Download Styled Image",
                    data=styled_image_io,
                    file_name="styled_image.jpg",
                    mime="image/jpeg"
                )
                st.success("Style transfer complete! 🎉")

        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
