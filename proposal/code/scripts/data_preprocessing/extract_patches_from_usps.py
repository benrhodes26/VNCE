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

parser = ArgumentParser(description='Extract non-overlapping 3*3 patches from the USPS dataset',
                        formatter_class=ArgumentDefaultsHelpFormatter)
# Read/write arguments
parser.add_argument('--data_dir', type=str, default='/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/data',
                    help='Path to directory where data is loaded and saved')
args = parser.parse_args()

rng = RandomState(70835236)

loaded = loadmat(os.path.join(args.data_dir, 'usps_all.mat'))
digits = loaded['data']

digits = digits.reshape(16, 16, -1)
digits = np.transpose(digits, [2, 1, 0])  # (11000, 16, 16)

# Binarise the images
ones = digits >= 150
zeros = digits < 150
digits[ones] = 1
digits[zeros] = 0

# Extract the 3*3 patches
patches = np.zeros((25*11000, 3, 3))
count = 0
for i in range(0, 15, 3):
    for j in range(0, 15, 3):
        patches[count*11000: (count+1)*11000, :, :] = digits[:, i:i+3, j:j+3]
        count += 1

# Calculate the empirical distribution over the data (discrete dist over 2^9=512 binary vectors)
patches = patches.reshape(25*11000, 9)
emp_dist = EmpiricalDist(patches, rng=rng)

# Shuffle data
perm = rng.permutation(np.arange(len(patches)))
patches = patches[perm]

# Split into train & test (80, 20)
split = int(len(patches)*0.8)
train_patches = patches[:split]
test_patches = patches[split:]

np.savez(os.path.join(args.data_dir, 'usps_3by3patches'), train=train_patches, test=test_patches)
pickle.dump(emp_dist, open(os.path.join(args.data_dir, 'usps_3by3_emp_dist.p'), 'wb'))
