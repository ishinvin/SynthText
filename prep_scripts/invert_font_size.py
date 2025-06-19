# Author: Ankush Gupta
# Date: 2015
"Script to generate font-models."

import pygame
from pygame import freetype
import numpy as np
import pickle as cp
import os

pygame.init()

ys = np.arange(8,200)
A = np.c_[ys,np.ones_like(ys)]

xs = []
models = {} #linear model

FONT_LIST = os.path.abspath(os.path.join(os.getcwd(), 'data/fonts/fontlist.txt'))
fonts = [os.path.join(os.getcwd(), 'data/fonts', f.strip()) for f in open(FONT_LIST)]
output_filename = os.path.abspath(os.path.join(os.getcwd(), 'data/models/font_px2pt.cp'))

for i in range(len(fonts)):
	print(i)
	font = freetype.Font(fonts[i], size=12)
	h = []
	for y in ys:
		h.append(font.get_sized_glyph_height(y))
	h = np.array(h)
	m,_,_,_ = np.linalg.lstsq(A,h)
	models[font.name] = m
	xs.append(h)

with open(output_filename,'wb') as f:
	cp.dump(models,f)
