import os
import random

random.seed(1)
# Define the directory path
dir_path = 'data/smallersets/real_sperm_data_biglabels_50+50/train2/'

images2keep = 50

# Get a list of all files in the directory
files_img = os.listdir(dir_path+'images')
files_lbl = os.listdir(dir_path+'labels')

files_img.sort()
files_lbl.sort()

# Generate a random vector of indices
random_vector = random.sample(range(len(files_img)), k=int(len(files_img)-images2keep))

# Remove files that match the indices in the random vector
for index in random_vector:
    file_path_img = os.path.join(dir_path+'images', files_img[index])
    if os.path.isfile(file_path_img):
        os.remove(file_path_img)
        print(f'Removed file: {file_path_img}')
    else:
        print(f'No such file: {file_path_img}')
    file_path_lbl = os.path.join(dir_path+'labels', files_lbl[index])
    if os.path.isfile(file_path_lbl):
        os.remove(file_path_lbl)
        print(f'Removed file: {file_path_lbl}')
    else:
        print(f'No such file: {file_path_lbl}')
