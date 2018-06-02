from matplotlib import pyplot as plt
import pickle
import os

dir = '/afs/inf.ed.ac.uk/user/s17/s1771906/masters-project/ben-rhodes-masters-project/experimental-results/mog/500_runs/sample_sizes'
fig2 = pickle.load(open(os.path.join(dir, 'fig2.p'), 'rb'))
ax = fig2.gca()
ax.grid()
fig2.savefig(os.path.join(dir, 'sample-size-against-mse-scaling-param2.pdf'))
