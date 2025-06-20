import pygame
import math
import numpy as np
import freetype
import uharfbuzz as hb
import pygame.locals
from font_state import FontState, BaselineState
from text_utils import crop_safe

# def shape_khmer(text, font_path):
#     face = hb.Face(hb.Blob.from_file_path(font_path))
#     font = hb.Font(face)

#     buf = hb.Buffer()
#     buf.add_str(text)
#     buf.guess_segment_properties()

#     hb.shape(font, buf)
#     infos = buf.glyph_infos

#     # HarfBuzz reshapes the text but returns glyph IDs; we need reordered characters
#     clusters = [(info.cluster, text[info.cluster]) for info in infos]
#     clusters.sort()  # reorder by cluster
#     shaped_text = ''.join([c[1] for c in clusters])

#     return shaped_text



# def render_curved(font, word_text, font_path):
#         """
#         use curved baseline for rendering word
#         """
#         word_text = word_text.replace('\u200c', ' ')
#         word_text = shape_khmer(word_text, font_path)
#         wl = len(word_text)

#         # create the surface:
#         lspace = font.get_sized_height() + 1
#         lbound = font.get_rect(word_text)
#         fsize = (round(2.0*lbound.width), round(3*lspace))
#         surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)

#         # baseline state
#         mid_idx = wl//2
#         BS = BaselineState().get_sample()
#         curve = [BS['curve'](i-mid_idx) for i in range(wl)]
#         curve[mid_idx] = -np.sum(curve) / (wl-1)
#         rots  = [-int(math.degrees(math.atan(BS['diff'](i-mid_idx)/(font.size/2)))) for i in range(wl)]

#         bbs = []
#         # place middle char
#         rect = font.get_rect(word_text)
#         rect.centerx = surf.get_rect().centerx
#         rect.centery = surf.get_rect().centery + rect.height
#         rect.centery +=  curve[mid_idx]
#         ch_bounds = font.render_to(surf, rect, word_text, rotation=rots[mid_idx])
#         ch_bounds.x = rect.x + ch_bounds.x
#         ch_bounds.y = rect.y - ch_bounds.y
#         mid_ch_bb = np.array(ch_bounds)

#         # render chars to the left and right:
#         ch_idx = []
#         bbs.append(mid_ch_bb)
#         ch_idx.append(0)

#         # correct the bounding-box order:
#         bbs_sequence_order = [None for i in ch_idx]
#         for idx,i in enumerate(ch_idx):
#             bbs_sequence_order[i] = bbs[idx]
#         bbs = bbs_sequence_order

#         # get the union of characters for cropping:
#         r0 = pygame.Rect(bbs[0])
#         rect_union = r0.unionall(bbs)

#         # crop the surface to fit the text:
#         bbs = np.array(bbs)
#         surf_arr, bbs = crop_safe(pygame.surfarray.pixels_alpha(surf), rect_union, bbs, pad=5)
#         surf_arr = surf_arr.swapaxes(0,1)
#         return surf_arr, word_text, bbs

def shape_text(text, fontfile, font_size):
    # Load font file bytes
    # with open(fontfile, 'rb') as f:
    #     fontdata = f.read()

    face = freetype.Face(fontfile)
    face.set_char_size(font_size * 64)

    hb_blob = hb.Blob.from_file_path(fontfile)
    hb_face = hb.Face(hb_blob, 0)
    hb_font = hb.Font(hb_face)
    hb_font.scale = (face.size.ascender, face.size.ascender)

    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(hb_font, buf)

    infos = buf.glyph_infos
    positions = buf.glyph_positions

    glyph_data = []
    for info, pos in zip(infos, positions):
        glyph_data.append({
            'glyph_id': info.codepoint,
            'cluster': info.cluster,
            'x_advance': pos.x_advance / 64,
            'y_advance': pos.y_advance / 64,
            'x_offset': pos.x_offset / 64,
            'y_offset': pos.y_offset / 64,
        })
    return face, glyph_data

def render_curved(text, fontfile, font_size):
    pygame.init()

    face, glyphs = shape_text(text, fontfile, font_size)

    # Calculate surface size (sum advances + padding)
    total_advance = int(sum(g['x_advance'] for g in glyphs)) + 100
    surf_height = font_size * 4
    surface = pygame.Surface((total_advance, surf_height), pygame.SRCALPHA)

    mid_idx = len(glyphs) // 2
    BS = BaselineState().get_sample()

    # Calculate curve offsets and rotations for each glyph
    curve = [BS['curve'](i - mid_idx) for i in range(len(glyphs))]
    curve[mid_idx] = -np.sum(curve) / (len(glyphs) - 1 if len(glyphs) > 1 else 1)
    rots = [-int(math.degrees(math.atan(BS['diff'](i - mid_idx) / (font_size / 2)))) for i in range(len(glyphs))]

    pen_x = 50  # start with some padding
    pen_y = surf_height // 2

    bbs = []

    for i, g in enumerate(glyphs):
        face.load_glyph(g['glyph_id'], freetype.FT_LOAD_RENDER | freetype.FT_LOAD_FORCE_AUTOHINT)

        bitmap = face.glyph.bitmap
        width, rows = bitmap.width, bitmap.rows

        if width == 0 or rows == 0:
            pen_x += g['x_advance']
            continue  # skip empty glyphs (spaces etc)

        # Create pygame surface from bitmap
        glyph_surf = pygame.Surface((width, rows), pygame.SRCALPHA)
        arr = np.array(bitmap.buffer, dtype=np.uint8).reshape(rows, width).T  # transpose for pygame surface
        pygame.surfarray.pixels_alpha(glyph_surf)[:, :] = arr

        # Apply vertical offset from HarfBuzz
        x = pen_x + g['x_offset'] + face.glyph.bitmap_left
        y = pen_y - face.glyph.bitmap_top + g['y_offset'] + curve[i]

        # Rotate glyph surface according to baseline curve
        rotated = pygame.transform.rotate(glyph_surf, rots[i])
        rect = rotated.get_rect(center=(x, y))

        surface.blit(rotated, rect.topleft)

        bbs.append(rect)
        pen_x += g['x_advance']

    return surface, bbs


def main():
    # fs = FontState()
    # font_sample = fs.sample()
    # font = fs.init_font(font_sample)

    surf_arr, bbs = render_curved("អាវុធយុទ្ធភណ្ឌ", '/Users/ishinvin/Documents/workspace/personal/python/SynthText/data/fonts/moul/Moul-Regular.ttf', 64)
    # surf = pygame.surfarray.make_surface(surf_arr.swapaxes(0,1))
    pygame.image.save(surf_arr, "khmertext.png")

if __name__ == "__main__":
    main()