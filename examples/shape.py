import uharfbuzz as hb
import freetype
from PIL import Image, ImageDraw

# Input text and font
text = "សួស្តី ពិភពលោក"  # "Hello world" in Khmer
font_path = "/usr/share/fonts/truetype/noto/NotoSansKhmer-Regular.ttf"

# Load font with FreeType
face = freetype.Face(font_path)
font_size = 48
face.set_char_size(font_size * 64)

# Load font data for HarfBuzz
with open(font_path, "rb") as f:
    font_data = f.read()

# HarfBuzz setup
hb_blob = hb.Blob(font_data)
hb_face = hb.Face(hb_blob, 0)
hb_font = hb.Font(hb_face)
hb_font.scale = (face.size.ascender, face.size.ascender)
hb_font.set_funcs(hb.FontFuncs.create())

# HarfBuzz shaping
buf = hb.Buffer()
buf.add_str(text)
buf.guess_segment_properties()
hb.shape(hb_font, buf)

infos = buf.glyph_infos
positions = buf.glyph_positions

# Estimate image size
width = sum(pos.x_advance for pos in positions) // 64 + 20
height = font_size + 20

# Create image
image = Image.new("L", (width, height), color=255)
draw = ImageDraw.Draw(image)

# Render glyphs
pen_x, pen_y = 10, 10 + face.size.ascender // 64

for info, pos in zip(infos, positions):
    gid = info.codepoint
    face.load_glyph(gid, freetype.FT_LOAD_RENDER)
    bitmap = face.glyph.bitmap
    top = face.glyph.bitmap_top
    left = face.glyph.bitmap_left

    glyph_img = Image.frombytes("L", (bitmap.width, bitmap.rows), bitmap.buffer)
    image.paste(glyph_img, (pen_x + left, pen_y - top))

    pen_x += pos.x_advance // 64
    pen_y += pos.y_advance // 64

# Save image
image = image.convert("RGB")
image.save("khmer_rendered.png")
print("Rendered image saved as khmer_rendered.png")
