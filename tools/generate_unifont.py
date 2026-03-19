"""
generate_unifont.py
--------------------
Generates the unifont.pickle file required by ScribeSynth's content encoder.
Each character in the model vocabulary is rendered as a 16x16 binary bitmap
using PIL's built-in default font (bypassing unstable internet downloads).

Usage:
    python tools/generate_unifont.py

Output:
    data/unifont.pickle
"""

import os
import pickle
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Vocabulary from dataset/loader.py  (must match exactly)
LETTERS = '_Only thewigsofrcvdampbkuq.A-210xT5\'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%'

GLYPH_SIZE   = 16   # model expects 16x16
OUTPUT_PATH  = 'data/unifont.pickle'

def render_glyph(char, font):
    """Render a single character as a GLYPH_SIZE x GLYPH_SIZE binary image."""
    img = Image.new('L', (GLYPH_SIZE * 4, GLYPH_SIZE * 4), color=255)
    draw = ImageDraw.Draw(img)
    draw.text((GLYPH_SIZE, GLYPH_SIZE), char, font=font, fill=0)

    # Centre-crop to GLYPH_SIZE x GLYPH_SIZE
    bbox = img.getbbox()
    if bbox:
        char_img = img.crop(bbox)
        char_img = char_img.resize((GLYPH_SIZE, GLYPH_SIZE), Image.LANCZOS)
    else:
        # Blank glyph (e.g. space)
        char_img = Image.new('L', (GLYPH_SIZE, GLYPH_SIZE), color=255)

    arr = np.array(char_img, dtype=np.float32) / 255.0   # [0, 1]
    arr = (arr < 0.5).astype(np.float32)                 # binarise: 1 = ink, 0 = bg
    return arr


def main():
    os.makedirs('data', exist_ok=True)
    
    # Use standard built-in font
    font = ImageFont.load_default()

    print(f'Rendering {len(LETTERS)} glyphs ...')
    symbols = []
    for char in LETTERS:
        mat = render_glyph(char, font)
        symbols.append({
            'idx': [ord(char)],
            'mat': mat.reshape(GLYPH_SIZE, GLYPH_SIZE),
        })

    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(symbols, f)

    print(f'Saved {len(symbols)} glyphs to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
