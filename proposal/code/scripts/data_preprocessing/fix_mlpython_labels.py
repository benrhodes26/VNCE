import numpy as np

load_dir = '~/masters-project/ben-rhodes-masters-project/proposal/data'
load_dir = os.path.expanduser(load_dir)
save_dir = '~/masters-project/ben-rhodes-masters-project/proposal/new_data'
save_dir = os.path.expanduser(save_dir)

for i, file in enumerate(os.listdir(load_dir)):
    load_path = os.path.join(load_dir, file)
    save_path = os.path.join(save_dir, file)

    data_dict = np.load(open(load_path, 'rb'), encoding="bytes")
    print('unfinished script')
    raise NotImplementedError