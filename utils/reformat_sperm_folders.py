import os
import shutil

folder_path = "data/videos_completos_detecciones"
dst_folder = "data/real_sperm_data"
img_path = os.path.join(dst_folder, "images")
label_path = os.path.join(dst_folder, "labels")

os.makedirs(img_path, exist_ok=True)
os.makedirs(label_path, exist_ok=True)

for path, directories, files in os.walk(folder_path):
    for f in files:
        if f.endswith('png'):
            name = path.split('/')[-1]
            name = name + "_" + f

            src = os.path.join(path, f)
            dst = os.path.join(img_path, name)

            shutil.copyfile(src, dst)
        elif f.endswith('txt'):
            name = path.split('/')[-1]
            name = name + "_" + f

            src = os.path.join(path, f)
            dst = os.path.join(label_path, name)
            shutil.copyfile(src, dst)
