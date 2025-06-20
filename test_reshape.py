import pygame
import math
import numpy as np
import freetype
import uharfbuzz as hb
from font_state import FontState, BaselineState
from text_utils import crop_safe

pygame.init()

import uharfbuzz as hb

def shape_khmer(text, font_path):
    with open(font_path, 'rb') as f:
        fontdata = f.read()

    face = hb.Face(hb.Blob(fontdata))
    font = hb.Font(face)

    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()

    hb.shape(font, buf)
    infos = buf.glyph_infos

    # HarfBuzz reshapes the text but returns glyph IDs; we need reordered characters
    clusters = [(info.cluster, text[info.cluster]) for info in infos]
    clusters.sort()  # reorder by cluster
    shaped_text = ''.join([c[1] for c in clusters])

    return shaped_text



def render_curved(font, word_text):
        """
        use curved baseline for rendering word
        """
        word_text = word_text.replace('\u200c', ' ')
        word_text = shape_khmer(word_text, "/Users/ishinvin/Documents/workspace/personal/python/SynthText/data/fonts/moul/Moul-Regular.ttf")
        wl = len(word_text)

        # create the surface:
        lspace = font.get_sized_height() + 1
        lbound = font.get_rect(word_text)
        fsize = (round(2.0*lbound.width), round(3*lspace))
        surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)

        # baseline state
        mid_idx = wl//2
        BS = BaselineState().get_sample()
        curve = [BS['curve'](i-mid_idx) for i in range(wl)]
        curve[mid_idx] = -np.sum(curve) / (wl-1)
        rots  = [-int(math.degrees(math.atan(BS['diff'](i-mid_idx)/(font.size/2)))) for i in range(wl)]

        bbs = []
        # place middle char
        rect = font.get_rect(word_text)
        rect.centerx = surf.get_rect().centerx
        rect.centery = surf.get_rect().centery + rect.height
        rect.centery +=  curve[mid_idx]
        ch_bounds = font.render_to(surf, rect, word_text, rotation=rots[mid_idx])
        ch_bounds.x = rect.x + ch_bounds.x
        ch_bounds.y = rect.y - ch_bounds.y
        mid_ch_bb = np.array(ch_bounds)

        # render chars to the left and right:
        ch_idx = []
        bbs.append(mid_ch_bb)
        ch_idx.append(0)

        # correct the bounding-box order:
        bbs_sequence_order = [None for i in ch_idx]
        for idx,i in enumerate(ch_idx):
            bbs_sequence_order[i] = bbs[idx]
        bbs = bbs_sequence_order

        # get the union of characters for cropping:
        r0 = pygame.Rect(bbs[0])
        rect_union = r0.unionall(bbs)

        # crop the surface to fit the text:
        bbs = np.array(bbs)
        surf_arr, bbs = crop_safe(pygame.surfarray.pixels_alpha(surf), rect_union, bbs, pad=5)
        surf_arr = surf_arr.swapaxes(0,1)
        return surf_arr, word_text, bbs


def main():
    fs = FontState()
    font = fs.sample()
    font = fs.init_font(font)

    surf_arr, text, bbs = render_curved(font, "អឹមអៀមស្ងៀមជាជាងស្ដី ស៊ីចេកខ្ចីជាជាងនៅមាត់ទទេ")
    surf = pygame.surfarray.make_surface(surf_arr.swapaxes(0,1))
    pygame.image.save(surf, "khmertext.png")

if __name__ == "__main__":
    main()