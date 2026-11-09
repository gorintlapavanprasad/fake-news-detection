"""
Draw the social preview image (1200 x 630) used for link previews: docs/preview.png.
Same visual language as the site: paper background, serif text, red/green pen marks, a stamp.

    python make_preview.py
Needs Pillow and the Windows fonts Georgia and Consolas (edit FONT_* for other systems).
"""

import os

from PIL import Image, ImageDraw, ImageFont

FONT_SERIF = r"C:\Windows\Fonts\georgia.ttf"
FONT_SERIF_B = r"C:\Windows\Fonts\georgiab.ttf"
FONT_MONO_B = r"C:\Windows\Fonts\consolab.ttf"
FONT_MONO = r"C:\Windows\Fonts\consola.ttf"

PAPER, INK, RULE, MUTE = (246, 247, 245), (28, 32, 36), (207, 211, 204), (93, 101, 96)
FAKE, REAL = (168, 40, 31), (29, 107, 67)


def blend(fg, alpha):
    return tuple(int(PAPER[i] * (1 - alpha) + fg[i] * alpha) for i in range(3))


def main():
    W, H = 1200, 630
    img = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(img)
    f = lambda p, s: ImageFont.truetype(p, s)

    d.text((70, 52), "Fake News Detector", font=f(FONT_SERIF_B, 78), fill=INK)
    d.text((70, 152), "Paste an article. See which words made it look real or fake.",
           font=f(FONT_SERIF, 32), fill=MUTE)

    # marked-up sample text (words and their pen level: negative = red, positive = green)
    box = (70, 225, 1130, 455)
    d.rectangle(box, outline=RULE, width=2)
    words = [("You", -2), ("will", 0), ("not", -1), ("believe", 0), ("what", -1), ("they", -2),
             ("are", -1), ("hiding!", 0), ("Insiders", 0), ("reveal", -2), ("the", -1),
             ("shadowy", 0), ("elite", 1), ("use", 0), ("machines", 0), ("to", 0),
             ("control", -1), ("storms.", 1), ("Share", -3), ("this", -2), ("before", 1),
             ("it", 0), ("gets", 0), ("deleted", 1), ("by", -1), ("the", -2),
             ("mainstream", -3), ("media!!!", -2)]
    font = f(FONT_SERIF, 40)
    x, y, pad_x, line_h = box[0] + 28, box[1] + 24, box[0] + 28, 64
    space = d.textlength(" ", font=font)
    for w, lvl in words:
        wl = d.textlength(w, font=font)
        if x + wl > box[2] - 28:
            x, y = pad_x, y + line_h
        if lvl:
            col = FAKE if lvl < 0 else REAL
            a = {1: 0.16, 2: 0.32, 3: 0.55}[abs(lvl)]
            d.rectangle((x - 3, y + 2, x + wl + 3, y + 52), fill=blend(col, a))
        d.text((x, y), w, font=font, fill=INK)
        x += wl + space

    # stamp, rotated slightly like a rubber stamp
    sf = f(FONT_MONO_B, 64)
    tw = int(d.textlength("FAKE", font=sf))
    stamp = Image.new("RGBA", (tw + 70, 110), (0, 0, 0, 0))
    sd = ImageDraw.Draw(stamp)
    sd.rectangle((4, 4, stamp.width - 5, stamp.height - 5), outline=FAKE + (255,), width=6)
    sd.text((35, 16), "FAKE", font=sf, fill=FAKE + (255,))
    stamp = stamp.rotate(2, resample=Image.BICUBIC, expand=True)
    img.paste(stamp, (70, 478), stamp)

    # number line with marker
    lx0, lx1, ly = 400, 1130, 560
    d.line((lx0, ly, lx1, ly), fill=INK, width=3)
    for tx in (lx0, (lx0 + lx1) // 2, lx1):
        d.line((tx, ly, tx, ly + 14), fill=INK, width=3)
    mx = lx0 + int((lx1 - lx0) * 0.02)
    d.polygon([(mx, ly - 2), (mx - 13, ly - 26), (mx + 13, ly - 26)], fill=INK)
    sm = f(FONT_MONO, 22)
    d.text((lx0, ly + 22), "fake", font=sm, fill=MUTE)
    d.text(((lx0 + lx1) // 2 - 28, ly + 22), "50/50", font=sm, fill=MUTE)
    d.text((lx1 - 50, ly + 22), "real", font=sm, fill=MUTE)
    d.text((400, 480), "Open source. Runs in your browser. Nothing is uploaded.", font=f(FONT_SERIF, 27), fill=MUTE)

    out = os.path.join(os.path.dirname(__file__), "..", "docs", "preview.png")
    img.save(out, optimize=True)
    print("saved", os.path.abspath(out), os.path.getsize(out) // 1024, "KB")


if __name__ == "__main__":
    main()
