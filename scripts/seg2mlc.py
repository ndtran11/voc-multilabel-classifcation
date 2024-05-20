import os
from argparse import ArgumentParser
from PIL import Image
import numpy as np
import shutil
import pandas as pd

def parse_opt():
    parser = ArgumentParser() 
    parser.add_argument('--img-dir', type=str, required=True)
    parser.add_argument('--mask-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--classnames-file', type=str, default=None)
    return parser.parse_args()

def main():
    opt = parse_opt()
    
    images = sorted(os.listdir(opt.img_dir))
    masks = [os.path.join(opt.mask_dir, f) for f in sorted(os.listdir(opt.mask_dir))]

    shutil.copytree(opt.img_dir, os.path.join(opt.output_dir, 'images'))

    item_labels = [] 
    all_labels = set([])
    
    for img_file, mask_file in zip(images, masks):
        # img = Image.open(img_file)
        mask = Image.open(mask_file)
        mask = np.array(mask)
        un = np.unique(mask)
        un = np.array([x for x in un if x != 255], dtype=np.int32)
        item_labels.append(un)
        all_labels.update(un)

    n_classes = len(all_labels)
    npy_labels = np.zeros((len(item_labels), n_classes), dtype=np.int32)

    for i, labels in enumerate(item_labels):
        npy_labels[i, labels] = 1
        
    np.save(os.path.join(opt.output_dir, 'labels.npy'), npy_labels)

    if opt.classnames_file is not None:
        shutil.copy(opt.classnames_file, os.path.join(opt.output_dir, 'classnames.txt'))

if __name__ == '__main__':
    main()