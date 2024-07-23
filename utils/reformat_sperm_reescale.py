import os
import shutil
import pandas as pd
import numpy as np

folder_path = "data/real_test_500x500/labels"
dst_path = "data/real_test_500x500/labels"
os.makedirs(dst_path, exist_ok=True)

for path, directories, files in os.walk(folder_path):
    for f in files:
        if f.endswith('txt'):
            df = pd.read_csv(os.path.join(path, f), sep=" ", header=None)
            df.columns = ["class", "x", "y", "w", "h"]
            df['w'] = 0.015 #* 1.35
            df['h'] = 0.015 #* 1.35

            df.to_csv(os.path.join(dst_path, f), sep=' ', index=False, header=False)

print(df.mean())
print(df.std())
