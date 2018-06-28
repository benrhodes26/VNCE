"""Script to extract non-overlapping 3*3 patches from the USPS dataset"""
import sys
code_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/code'
if code_dir not in sys.path:
    sys.path.append(code_dir)

import numpy as np
import os
import pickle

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from matplotlib import pyplot as plt
from numpy.random import RandomState
from scipy.io import loadmat

from distribution import EmpiricalDist

parser = ArgumentParser(description='Extract non-overlapping patches from the USPS dataset',
                        formatter_class=ArgumentDefaultsHelpFormatter)
# Read/write arguments
parser.add_argument('--data_dir', type=str, default='/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/data/usps',
                    help='Path to directory where data is loaded and saved')
parser.add_argument('--patch_width', type=int, default=4, help='')
parser.add_argument('--patch_height', type=int, default=4, help='')
args = parser.parse_args()

rng = RandomState(70835236)

test_train_split = 0.8
num_images = 11000
image_dim = 16

width = args.patch_width
height = args.patch_height
num_patches_wide = int(image_dim / width)
num_patches_high = int(image_dim / height)
num_patches_per_image = num_patches_wide * num_patches_high

loaded = loadmat(os.path.join(args.data_dir, 'usps_all.mat'))
digits = loaded['data']

digits = digits.reshape(image_dim, image_dim, -1)
digits = np.transpose(digits, [2, 1, 0])  # (11000, 16, 16)

# Binarise the images. Thresholding at 150 produces legible binary digits.
ones = digits >= 150
zeros = digits < 150
digits[ones] = 1
digits[zeros] = 0

# Extract the width*height patches
patches = np.zeros((num_patches_per_image*num_images, width, height))
count = 0
for i in range(0, num_patches_wide*width, width):
    for j in range(0, num_patches_high*height, height):
        patches[count*num_images: (count+1)*num_images, :, :] = digits[:, i:i+width, j:j+height]
        count += 1
print('counted {} patches per image'.format(count))

# remove empty patches. Do this by getting the indices of the patches that contain a non-zero value
mask = np.any(patches, axis=(1, 2))
patches = patches[mask]

# Calculate the empirical distribution over the data (if the dimensionality is low, otherwise the data is too sparse)
patches = patches.reshape(-1, width*height)
if width * height <= 12:
    emp_dist = EmpiricalDist(patches, rng=rng)

# Shuffle data
perm = rng.permutation(np.arange(len(patches)))
patches = patches[perm]

# Split into train & test (80, 20)
split = int(len(patches)*test_train_split)
train_patches = patches[:split]
test_patches = patches[split:]

np.savez(os.path.join(args.data_dir, 'usps_{}by{}patches'.format(width, height)), train=train_patches, test=test_patches)
if width * height <= 12:
    pickle.dump(emp_dist, open(os.path.join(args.data_dir, 'usps_{}by{}_emp_dist.p'.format(width, height)), 'wb'))

print('there are {} non-empty image patches'.format(len(patches)))
