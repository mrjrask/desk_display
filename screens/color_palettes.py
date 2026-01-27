# color_palettes.py

import random

# Complementary color palettes for each screen type
SCREEN_PALETTES = {
    'date_time':      [(255, 0, 0),    (0, 255, 0),    (0, 0, 255),    (255, 255, 0)],
    'weather':        [(0, 128, 255),  (255, 128, 0)],
    'sports':         [(220, 20, 60),  (30, 144, 255)],
    'travel_time':    [(34, 139, 34),  (255, 215, 0)],
    'vrnof':          [(128, 0, 128),  (0, 255, 255)],
    'logo_scroll':    [(255, 165, 0),  (75, 0, 130)],
}

def get_palette(screen: str):
    """Return the palette list for a given screen key (or white fallback)."""
    return SCREEN_PALETTES.get(screen, [(255, 255, 255)])

def random_color(screen: str):
    """Pick one color at random from that screen’s palette."""
    return random.choice(get_palette(screen))
