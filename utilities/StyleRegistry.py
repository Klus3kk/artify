import random

class StyleRegistry:
    def __init__(self):
        self.styles = {
            "impressionism": [
                "images/style/impressionism/water_lilies.jpg",
                "images/style/impressionism/ballet_rehearsal.jpg"
            ],
            "abstract": [
                "images/style/abstract/composition_viii.jpg",
                "images/style/abstract/autumn_rhythm.jpg"
            ],
            "cubism": [
                "images/style/cubism/les_demoiselles.jpg",
                "images/style/cubism/violin_and_candlestick.jpg"
            ]
        }

    def get_random_style_image(self, category):
        if category in self.styles:
            return random.choice(self.styles[category])
        raise ValueError(f"Style category '{category}' not found!")
