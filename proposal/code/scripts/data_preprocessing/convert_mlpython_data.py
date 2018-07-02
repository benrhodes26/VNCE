import mlpython.datasets.adult as a
import mlpython.datasets.connect4 as c
import mlpython.datasets.dna as d
import mlpython.datasets.mushrooms as m
import mlpython.datasets.nips as n
import mlpython.datasets.ocr_letters as o
import mlpython.datasets.rcv1 as r
import mlpython.datasets.web as w

import os
import numpy as np

load_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/mlpython/data'
adult = a.load(load_dir + '/adult', True)
connect4 = c.load(load_dir + '/connect4', True)
dna = d.load(load_dir + '/dna', True)
mushrooms = m.load(load_dir + '/mushrooms', True)
nips = n.load(load_dir + '/nips', True)
ocr_letters = o.load(load_dir + '/ocr_letters', True)
rcv1 = r.load(load_dir + '/rcv1', True)
web = w.load(load_dir + '/web', True)

dataset_names = ["adult", "connect4", "dna", "mushrooms", "nips", "ocr_letters", "rcv1", "web"]
datasets = [adult, connect4, dna, mushrooms, nips, ocr_letters, rcv1, web]
modified_datasets = []
for i, dataset in enumerate(datasets):
    modified_datasets.append({})
    for key, val in dataset.items():
        val = val[0]  # either train, val or test (val[1] is associated metadata, which we ignore)
        modified_datasets[-1][key] = np.array([datapoint for datapoint in val])

save_dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/proposal/data'
for i, dataset in enumerate(modified_datasets):
    np.savez(os.path.join(save_dir, dataset_names[i]), **dataset)
