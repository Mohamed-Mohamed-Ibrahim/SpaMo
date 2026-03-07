import os
import numpy as np
import torch
from tqdm import tqdm

dirs = ['feat_Phoenix14T', 'mae_feat_Phoenix14T']

files = []
for d in dirs:
    for root, _, fs in os.walk(d):
        for f in fs:
            if f.endswith('.npy') or f.endswith('.pt'):
                files.append(os.path.join(root, f))

print("Files found:", len(files))

for path in tqdm(files):

    try:
        if path.endswith(".npy"):

            arr = np.load(path)

            # FORCE full load
            arr = np.array(arr)

            if np.isnan(arr).any():
                print("NaN detected:", path)

            if np.isinf(arr).any():
                print("Inf detected:", path)

            if arr.dtype != np.float32:
                print("Unexpected dtype:", path, arr.dtype)

            if arr.ndim != 2:
                print("Unexpected shape:", path, arr.shape)

            if arr.shape[0] > 5000:
                print("Very long sequence:", path, arr.shape)

        else:

            obj = torch.load(path, map_location='cpu')

            if isinstance(obj, torch.Tensor):

                if torch.isnan(obj).any():
                    print("NaN tensor:", path)

                if torch.isinf(obj).any():
                    print("Inf tensor:", path)

    except Exception as e:

        print("BROKEN FILE:", path)
        print(e)