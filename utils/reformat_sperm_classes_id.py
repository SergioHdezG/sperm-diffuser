import os
import shutil
import pandas as pd

folder_path = "data/real_sperm_data/labels"

for path, directories, files in os.walk(folder_path):
    for f in files:
        if f.endswith('txt'):
            df = pd.read_csv(os.path.join(path, f), sep=" ", header=None)
            df.columns = ["class", "x", "y", "w", "h"]
            df['class'] = 0

            df.to_csv(os.path.join(path, f), sep=' ', index=False, header=False)
