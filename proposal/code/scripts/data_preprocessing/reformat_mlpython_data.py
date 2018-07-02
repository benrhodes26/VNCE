import numpy as np
import os

load_dir = '~/masters-project/ben-rhodes-masters-project/proposal/data'
load_dir = os.path.expanduser(load_dir)
save_dir = '~/masters-project/ben-rhodes-masters-project/proposal/new_data'
save_dir = os.path.expanduser(save_dir)

for i, file in enumerate(os.listdir(load_dir)):
    load_path = os.path.join(load_dir, file)
    save_path = os.path.join(save_dir, file)

    data_dict = np.load(open(load_path, 'rb'), encoding="bytes")
    new_data_dict = {}
    for key, val in data_dict.items():
        data = val[:, 0]
        num_datapoints = len(data)
        new_data_array = np.concatenate(data).reshape(num_datapoints, -1)
        new_data_dict[key] = new_data_array

    np.savez(open(save_path, 'wb'), **new_data_dict)
    meta_data = ''.join(['{}: {} \n'.format(key, value.shape) for key, value in new_data_dict.items()])
    with open(os.path.join(save_dir, 'metadata.txt'), 'a') as f:
        f.write('{} \n {}'.format(file[:-4], meta_data))
