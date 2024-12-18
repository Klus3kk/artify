import random

class StyleRegistry:
    def __init__(self):
        self.styles = {
            "abstract": [
                "images/style/abstract/Frankenthaler_Helen_Mountains_and_Sea.jpg",
                "images/style/abstract/Jackson_Pollock_Autumn_Rhythm.jpg",
                "images/style/abstract/Jackson_Pollock_Full-Fathom-Five.jpg",
                "images/style/abstract/Joan_Miro_Triptych_Bleu.jpg",
                "images/style/abstract/Paul_Klee_Senecio.jpg",
                "images/style/abstract/Piet_Mondrian_Broadway_Boogie_Woogie.jpg",
                "images/style/abstract/Theo_van_Doesburg_Peinture_Pure.jpg",
                "images/style/abstract/Vassily_Kandinsky_Composition_VIII.jpg"
            ],
            "baroque": [
                "images/style/baroque/Annibale_Carracci_Lamentation_of_Christ.jpg",
                "images/style/baroque/Caravaggio_Death_of_the_Virgin.jpg",
                "images/style/baroque/Caravaggio_Judith_Beheading_Holofernes.jpg",
                "images/style/baroque/Caravaggio_The_Calling_of_Saint_Matthew.jpg",
                "images/style/baroque/Diego_Velázquez_Las_Meninas.jpg",
                "images/style/baroque/Johannes_Vermeer_Girl_with_a_Pearl_Earring.jpg",
                "images/style/baroque/Rembrandt_The_Anatomy_Lesson_of_Dr_Nicolaes_Tulp.jpg",
                "images/style/baroque/Rembrandt_van_Rijn_The_Night_Watch.jpg"
            ],
            "cubism": [
                "images/style/cubism/Georges_Braque_Natura_morta_con clarinetto.jpg",
                "images/style/cubism/Georges_Braque_Violin_and_Candlestick.jpg",
                "images/style/cubism/Juan_Gris_Guitar_on_the_table.jpg",
                "images/style/cubism/Juan_Gris_Portrait_of_Pablo_Picasso.jpg",
                "images/style/cubism/Marc_Chagall_I_am_The_Village.jpg",
                "images/style/cubism/Pablo_Picasso_Guernica.jpg",
                "images/style/cubism/Pablo-Picasso-Panny-Z-Awinionu.jpg",
                "images/style/cubism/Paul_Cézanne_Les_Grandes_Baigneuses.jpg"
            ],
            "expressionism": [
                "images/style/expressionism/David_Alfaro_Siqueiros_Birth_of_Fascism.jpg",
                "images/style/expressionism/Edvard_Munch_The_Scream.jpg",
                "images/style/expressionism/Emil_Nolde_Dance_Around_the_Golden Calf.jpg",
                "images/style/expressionism/Ernst_Ludwig_Kirchner_Street_Berlin.jpg",
                "images/style/expressionism/Marc_Franz_Blaues_Pferdchen_Saarlandmuseum.jpg",
                "images/style/expressionism/Max_Beckmann_The_Night.jpg",
                "images/style/expressionism/Pablo_Picasso_Stary_Gitarzysta.jpg",
                "images/style/expressionism/Wassily_Kandinsky_The_Blue_Rider.jpg"
            ],
            "impressionism": [
                "images/style/impressionism/Auguste_Renoir_Dance_at_Le_Moulin_de_la_Galette.jpg",
                "images/style/impressionism/Claude_Monet_Impression_Sunrise.jpg",
                "images/style/impressionism/Claude_Monet_Lilies.jpg",
                "images/style/impressionism/Claude_Monet_Woman_with_a_Parasol.jpg",
                "images/style/impressionism/Edgar_Degas_The_Dance_Class.jpg",
                "images/style/impressionism/Edouard_Manet_Woman_Reading.jpg",
                "images/style/impressionism/Gustave_Caillebotte_LHomme_au_balcon_boulevard.jpg",
                "images/style/impressionism/Gustave_Caillebotte_Paris_Street_Rainy_Day.jpg"
            ]
        }

    def get_random_style_image(self, category):
        if category in self.styles:
            return random.choice(self.styles[category])
        raise ValueError(f"Style category '{category}' not found!")

