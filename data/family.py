import os
from collections import defaultdict

import numpy as np
from PIL import Image

def load(path):
    with Image.open(path) as im:
        im = im.convert('RGB')
        X = np.array(im).transpose((2, 0, 1))
        return (X / 255.0).astype('float32')

def imgname2bag(filename):
    return filename.split("-")[0]

def imgname2label(filename):
    label = filename[filename.rfind('.') - 1]
    assert label in ['f', 'm']
    return 1 if label == 'f' else 0

def group_bags(filenames):
    bags = defaultdict(list)
    for f in filenames:
        bags[imgname2bag(f)].append(f)
    return list(bags.values())

def family_dataset(path):
    filenames = os.listdir(path)
    groups = group_bags(filenames)

    bags = []
    for group in groups:
        X = np.stack([load(os.path.join(path, n)) for n in group])
        t = np.array([imgname2label(n) for n in group], 'float32')
        y = np.sum(t)
        bags.append((X, y, t))

    Xs, ys, ts = zip(*bags)

    return Xs, ys, ts

#Xs, ys, ts = family_dataset("images")
#print(len(Xs), len(ys), len(ts))

