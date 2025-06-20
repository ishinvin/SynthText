import os
import math
import pygame
import numpy as np
import scipy.signal as ssig
import pygame.locals

from text_source import TextSource
from font_state import FontState, BaselineState
from text_utils import crop_safe, move_bb

class RenderFont(object):
    """
    Outputs a rasterized font sample.
        Output is a binary mask matrix cropped closesly with the font.
        Also, outputs ground-truth bounding boxes and text string
    """

    def __init__(self, data_dir='data'):
        # distribution over the type of text:
        # whether to get a single word, paragraph or a line:

        ## TEXT PLACEMENT PARAMETERS:
        self.f_shrink = 0.90
        self.max_shrink_trials = 5 # 0.9^5 ~= 0.6
        # the minimum number of characters that should fit in a mask
        # to define the maximum font height.
        self.min_nchar = 2
        self.min_font_h = 16 #px : 0.6*12 ~ 7px <= actual minimum height
        self.max_font_h = 120 #px
        self.p_flat = 0.10

        # curved baseline:
        self.p_curved = 1.0
        self.baselinestate = BaselineState()

        # text-source : gets english text:
        self.text_source = TextSource(self.min_nchar, os.path.join(data_dir,'newsgroup/khmer_words.txt'))

        # get font-state object:
        self.font_state = FontState(data_dir)

        pygame.init()

    # def render_curved(self, font, word_text):
    #     """
    #     use curved baseline for rendering word
    #     """
    #     word_text = word_text.replace('\u200c', ' ')
    #     wl = len(word_text)

    #     # create the surface:
    #     lspace = font.get_sized_height() + 1
    #     lbound = font.get_rect(word_text)
    #     fsize = (round(2.0*lbound.width), round(3*lspace))
    #     surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)

    #     # baseline state
    #     mid_idx = wl//2
    #     BS = self.baselinestate.get_sample()
    #     curve = [BS['curve'](i-mid_idx) for i in range(wl)]
    #     curve[mid_idx] = -np.sum(curve) / (wl-1)
    #     rots  = [-int(math.degrees(math.atan(BS['diff'](i-mid_idx)/(font.size/2)))) for i in range(wl)]

    #     bbs = []
    #     # place middle char
    #     rect = font.get_rect(word_text)
    #     rect.centerx = surf.get_rect().centerx
    #     rect.centery = surf.get_rect().centery + rect.height
    #     rect.centery +=  curve[mid_idx]
    #     ch_bounds = font.render_to(surf, rect, word_text, rotation=rots[mid_idx])
    #     ch_bounds.x = rect.x + ch_bounds.x
    #     ch_bounds.y = rect.y - ch_bounds.y
    #     mid_ch_bb = np.array(ch_bounds)

    #     # render chars to the left and right:
    #     ch_idx = []
    #     bbs.append(mid_ch_bb)
    #     ch_idx.append(0)

    #     # correct the bounding-box order:
    #     bbs_sequence_order = [None for i in ch_idx]
    #     for idx,i in enumerate(ch_idx):
    #         bbs_sequence_order[i] = bbs[idx]
    #     bbs = bbs_sequence_order

    #     # get the union of characters for cropping:
    #     r0 = pygame.Rect(bbs[0])
    #     rect_union = r0.unionall(bbs)

    #     # crop the surface to fit the text:
    #     bbs = np.array(bbs)
    #     surf_arr, bbs = crop_safe(pygame.surfarray.pixels_alpha(surf), rect_union, bbs, pad=5)
    #     surf_arr = surf_arr.swapaxes(0,1)
    #     return surf_arr, word_text, bbs

    def render_curved(self, font, word_text):
        word_text = word_text.replace('\u200c', ' ')
        
        # font metrics
        lspace = font.get_sized_height() + 1
        lbound = font.get_rect(word_text)
        fsize = (round(2.0 * lbound.width), round(3 * lspace))
        surf = pygame.Surface(fsize, pygame.SRCALPHA, 32)
        surf = surf.convert_alpha()

        wl = len(word_text)
        mid_idx = wl // 2

        # Curved baseline sample
        BS = self.baselinestate.get_sample()
        curve = [BS['curve'](i - mid_idx) for i in range(wl)]
        curve[mid_idx] = -np.sum(curve) / (wl - 1)

        # Render entire word as one unit
        rect = font.get_rect(word_text)
        rect.centerx = surf.get_rect().centerx
        rect.centery = surf.get_rect().centery + rect.height
        rect.centery += curve[mid_idx]

        ch_bounds = font.render_to(surf, rect, word_text)
        ch_bounds.x = rect.x + ch_bounds.x
        ch_bounds.y = rect.y - ch_bounds.y

        bb = np.array(ch_bounds)

        # Get cropped result
        r0 = pygame.Rect(bb)
        surf_arr, bbs = crop_safe(pygame.surfarray.pixels_alpha(surf), r0, [bb], pad=5)
        surf_arr = surf_arr.swapaxes(0, 1)
        return surf_arr, word_text, bbs

    def get_nline_nchar(self,mask_size,font_height,font_width):
        """
        Returns the maximum number of lines and characters which can fit
        in the MASK_SIZED image.
        """
        H,W = mask_size
        nline = int(np.ceil(H/(2*font_height)))
        nchar = int(np.floor(W/font_width))
        return nline,nchar

    def place_text(self, text_arrs, back_arr, bbs):
        areas = [-np.prod(ta.shape) for ta in text_arrs]
        order = np.argsort(areas)

        locs = [None for i in range(len(text_arrs))]
        out_arr = np.zeros_like(back_arr)
        for i in order:            
            ba = np.clip(back_arr.copy().astype(float), 0, 255)
            ta = np.clip(text_arrs[i].copy().astype(float), 0, 255)
            ba[ba > 127] = 1e8
            intersect = ssig.fftconvolve(ba,ta[::-1,::-1],mode='valid')
            safemask = intersect < 1e8

            if not np.any(safemask): # no collision-free position:
                #warn("COLLISION!!!")
                return back_arr,locs[:i],bbs[:i],order[:i]

            minloc = np.transpose(np.nonzero(safemask))
            loc = minloc[np.random.choice(minloc.shape[0]),:]
            locs[i] = loc

            # update the bounding-boxes:
            bbs[i] = move_bb(bbs[i],loc[::-1])

            # blit the text onto the canvas
            w,h = text_arrs[i].shape
            out_arr[loc[0]:loc[0]+w,loc[1]:loc[1]+h] += text_arrs[i]

        return out_arr, locs, bbs, order

    def robust_HW(self,mask):
        m = mask.copy()
        m = (~mask).astype('float')/255
        rH = np.median(np.sum(m,axis=0))
        rW = np.median(np.sum(m,axis=1))
        return rH,rW

    def sample_font_height_px(self,h_min,h_max):
        if np.random.rand() < self.p_flat:
            rnd = np.random.rand()
        else:
            rnd = np.random.beta(2.0,2.0)

        h_range = h_max - h_min
        f_h = np.floor(h_min + h_range*rnd)
        return f_h

    def bb_xywh2coords(self,bbs):
        """
        Takes an nx4 bounding-box matrix specified in x,y,w,h
        format and outputs a 2x4xn bb-matrix, (4 vertices per bb).
        """
        n,_ = bbs.shape
        coords = np.zeros((2,4,n))
        for i in range(n):
            coords[:,:,i] = bbs[i,:2][:,None]
            coords[0,1,i] += bbs[i,2]
            coords[:,2,i] += bbs[i,2:4]
            coords[1,3,i] += bbs[i,3]
        return coords


    def render_sample(self,font,mask):
        """
        Places text in the "collision-free" region as indicated
        in the mask -- 255 for unsafe, 0 for safe.
        The text is rendered using FONT, the text content is TEXT.
        """
        #H,W = mask.shape
        H,W = self.robust_HW(mask)
        f_asp = self.font_state.get_aspect_ratio(font)

        # find the maximum height in pixels:
        max_font_h = min(0.9*H, (1/f_asp)*W/(self.min_nchar+1))
        max_font_h = min(max_font_h, self.max_font_h)
        if max_font_h < self.min_font_h: # not possible to place any text here
            return #None

        # let's just place one text-instance for now
        ## TODO : change this to allow multiple text instances?
        i = 0
        while i < self.max_shrink_trials and max_font_h > self.min_font_h:
            # if i > 0:
            #     print colorize(Color.BLUE, "shrinkage trial : %d"%i, True)

            # sample a random font-height:
            f_h_px = self.sample_font_height_px(self.min_font_h, max_font_h)
            #print "font-height : %.2f (min: %.2f, max: %.2f)"%(f_h_px, self.min_font_h,max_font_h)
            # convert from pixel-height to font-point-size:
            f_h = self.font_state.get_font_size(font, f_h_px)

            # update for the loop
            max_font_h = f_h_px 
            i += 1

            font.size = f_h # set the font-size

            # compute the max-number of lines/chars-per-line:
            nline,nchar = self.get_nline_nchar(mask.shape[:2],f_h,f_h*f_asp)
            #print "  > nline = %d, nchar = %d"%(nline, nchar)

            assert nline >= 1 and nchar >= self.min_nchar

            # sample text:
            text = self.text_source.sample_word(nline,nchar)
            if len(text)==0 or np.any([len(line)==0 for line in text]):
                continue
            #print colorize(Color.GREEN, text)

            # render the text:
            txt_arr,txt,bb = self.render_curved(font, text)
            bb = self.bb_xywh2coords(bb)

            # make sure that the text-array is not bigger than mask array:
            if np.any(np.r_[txt_arr.shape[:2]] > np.r_[mask.shape[:2]]):
                #warn("text-array is bigger than mask")
                continue

            # position the text within the mask:
            text_mask,loc,bb, _ = self.place_text([txt_arr], mask, [bb])
            if len(loc) > 0: #successful in placing the text collision-free:
                return text_mask,loc[0],bb[0],text
        return