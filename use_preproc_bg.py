"""
Sample code for load the 8000 pre-processed background image data.
Before running, first download the files from:
  https://github.com/ankush-me/SynthText#pre-generated-dataset
"""

import os
import h5py
import pickle
import traceback
import numpy as np

from PIL import Image
from synthgen import RendererV3
from common import colorize, Color
from utils import add_res_to_db

INSTANCE_PER_IMAGE = 1

base_dir = 'bg_data'
img_dir = os.path.join(base_dir, 'bg_img')
depth_db = h5py.File(os.path.join(base_dir, 'depth.h5'),'r')
seg_db = h5py.File(os.path.join(base_dir, 'seg.h5'),'r')

def main():
    if not os.path.exists('results'):
       os.makedirs('results')
       
    out_file = 'results/khmer-synth.h5'
    out_db = h5py.File(out_file, 'w')
    out_db.create_group('/data')
    print(colorize(Color.GREEN,'Storing the output in: '+ out_file, bold=True))

    RV3 = RendererV3('data', max_time=5)

    imnames = sorted(depth_db.keys())
    total_img = len(imnames)
    with open('bg_data/imnames.cp', 'rb') as f:
       filtered_imnames = set(pickle.load(f))

    for idx, imname in enumerate(imnames):
        # ignore if not in filetered list:
        if imname not in filtered_imnames: continue
        try:
            # get the colour image:
            img = Image.open(os.path.join(img_dir, imname)).convert('RGB')
            
            # get depth:
            depth = depth_db[imname][:].T
            depth = depth[:,:,0]
            
            # get segmentation info:
            seg = seg_db['mask'][imname][:].astype('float32')
            area = seg_db['mask'][imname].attrs['area']
            label = seg_db['mask'][imname].attrs['label']
            
            # re-size uniformly:
            sz = depth.shape[:2][::-1]
            img = np.array(img.resize(sz,Image.Resampling.LANCZOS))
            seg = np.array(Image.fromarray(seg).resize(sz,Image.Resampling.NEAREST))

            print (colorize(Color.RED,'%d of %d'%(idx + 1, total_img), bold=True))
            res = RV3.render_text(img, depth, seg, area, label, ninstance=INSTANCE_PER_IMAGE)
            
            if len(res) > 0:
                add_res_to_db(imname,res,out_db)
        except:
            traceback.print_exc()
            print (colorize(Color.GREEN,'>>>> CONTINUING....', bold=True))
            continue
    out_db.close()
    
if __name__=='__main__':
    main()