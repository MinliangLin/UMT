import numpy as np
from pathlib import Path
from time import time

for fs in ['/fsx', '/efs']:
    start = time()
    dir_ = fs + '/minliang/umt'
    files = sorted((dir_ / Path('data/hotstar_highlight_865/clip_features')).glob('*.npz'))
    for f in files:
        _ = np.load(f)
    print(fs, time()-start)
