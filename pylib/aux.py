import os
import numpy as np
import matplotlib.pyplot as plt

path = 'datasets/FulBezier2FulSpermMasked/trainSynthMask'

for path, directories, files in os.walk(path):
    for f in files:
        if f.endswith('.npy'):
            print(f)
            npy = np.load(os.path.join(path, f))
            img = np.asarray(npy * 255., dtype=np.uint8)
            img = np.stack((img,)*3, axis=-1)
            plt.imsave(os.path.join(path, f[:-4] + '.png'), img)
            os.remove(os.path.join(path, f))