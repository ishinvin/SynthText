# Author: Ankush Gupta
# Date: 2015
"Script to generate font-models."

import os
import pickle
import numpy as np
from pygame import freetype

freetype.init()

ys = np.arange(8,200)
A = np.c_[ys,np.ones_like(ys)]

xs = []
models = {}
fonts = [os.path.join('data/fonts', f.strip()) for f in open('data/fonts/fontlist.txt')]
output_filename = 'data/models/font_px2pt.cp'

for i in range(len(fonts)):
	print(i)
	font = freetype.Font(fonts[i], size=12)
	h = []
	for y in ys:
		h.append(font.get_sized_glyph_height(float(y)))
	h = np.array(h)
	m,_,_,_ = np.linalg.lstsq(A,h)
	models[font.name] = m
	xs.append(h)

with open(output_filename,'wb') as f:
	pickle.dump(models,f)
