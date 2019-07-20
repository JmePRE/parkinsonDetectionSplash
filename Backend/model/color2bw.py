from PIL import Image
import numpy as np
import os


def c2bw(fn):
    t = 30
    col = Image.open(fn)
    gray = col.convert('L')

    # Let numpy do the heavy lifting for converting pixels to pure black or white
    bw = np.asarray(gray).copy()
    gval = bw.mean(axis=0).mean()
    # Pixel range is 0...255, 256/2 = 128
    bw[bw < (gval-t)] = 0     # Black
    bw[bw >= (gval-t)] = 255  # White

    # Now we put it back in Pillow/PIL land
    imfile = Image.fromarray(bw)
    return imfile


def c2bwdir(dirname):
    dirname += '/'
    for filename in os.listdir(dirname):
        if filename.endswith(".png"):
            new_image = c2bw(dirname + filename)
            new_image.save(dirname + filename)


c2bwdir('training_data/testing/parkinson')
c2bwdir('training_data/training/parkinson')
c2bwdir('training_data/testing/healthy')
c2bwdir('training_data/training/healthy')
