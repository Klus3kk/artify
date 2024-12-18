import random
import os


class StyleRegistry:
    def __init__(self, styles_dir="images/style"):
        """
        Initialize the StyleRegistry and dynamically load styles from the directory.

        :param styles_dir: Path to the directory containing style categories and images.
        """
        self.styles_dir = styles_dir
        self.styles = self._load_styles()

    def _load_styles(self):
        """
        Load style categories and image paths dynamically from the directory.

        :return: A dictionary with style categories as keys and lists of image paths as values.
        """
        styles = {}
        for category in os.listdir(self.styles_dir):
            category_path = os.path.join(self.styles_dir, category)
            if os.path.isdir(category_path):
                styles[category] = [
                    os.path.join(category_path, file)
                    for file in os.listdir(category_path)
                    if file.endswith(('.jpg', '.png'))
                ]
        return styles

    def get_random_style_image(self, category):
        """
        Get a random image path from the specified style category.

        :param category: Style category name.
        :return: Path to a random image in the category.
        """
        if category in self.styles:
            return random.choice(self.styles[category])
        raise ValueError(f"Style category '{category}' not found!")
