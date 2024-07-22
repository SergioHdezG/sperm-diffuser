import ast
import json
import os
import random
import shutil

import cv2
import numpy as np

import matplotlib.pyplot as plt

from diffuser.environments.utils.Bezier import Bezier
import matplotlib as mpl
from scipy import signal
import pandas as pd

resize_img = (500,500)
only_one_class = True
save_B_set = False
random_test = False
percent_train = 1.0
verbose = 0
seed = 1
correction_box_size = [0.005, 0.005]
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)

original_shape = (1024, 1280)
video_size = (536, 350)

spline_path = 'diffuser/datasets/synth_dataset/json/field_{}/json_bezier_spline'

splines_names = [spline_path.format(str(i)) for i in range(0, 84)]

save_train = 'diffuser/datasets/synth_dataset/images'
save_train_labels_yolo = 'diffuser/datasets/synth_dataset/yolo_labels'
save_train_labels_complete_csv = 'diffuser/datasets/synth_dataset/complete_labels'

aux_path = save_train.replace('train', 'aux')
os.makedirs(aux_path, exist_ok=True)
os.makedirs(save_train_labels_yolo, exist_ok=True)
os.makedirs(save_train_labels_complete_csv, exist_ok=True)

font = cv2.FONT_HERSHEY_SIMPLEX

def read_json(path):
    with open(path, 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)
    json_object = ast.literal_eval(json_object)
    return json_object

def gaussian_heatmap(center = (2, 2), image_size = (10, 10), sig = 1):
    """
    It produces single gaussian at expected center
    :param center:  the mean position (X, Y) - where high value expected
    :param image_size: The total image size (width, height)
    :param sig: The sigma value
    :return:
    """
    x_axis = np.linspace(0, image_size[0]-1, image_size[0]) - center[0]
    y_axis = np.linspace(0, image_size[1]-1, image_size[1]) - center[1]
    xx, yy = np.meshgrid(x_axis, y_axis)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel

def paint_sperm(full_image, sperm_image, head_coord, sperm_img_shape):
    min_y = int(head_coord[1] - int(sperm_img_shape[0] / 2))
    max_y = int(head_coord[1] + int(sperm_img_shape[0] / 2))
    min_x = int(head_coord[0] - int(sperm_img_shape[1] / 2))
    max_x = int(head_coord[0] + int(sperm_img_shape[1] / 2))
    try:
        full_image[min_y:max_y, min_x:max_x, :] = \
            np.clip(full_image[min_y:max_y, min_x:max_x, :] + sperm_image, 0, 255)
        painted = True
    except:
        new_min_y = 0 if min_y >= 0 else np.abs(min_y)
        new_max_y = sperm_image.shape[0] if max_y <= full_image.shape[0] else -np.abs(full_image.shape[0] - max_y)
        new_min_x = 0 if min_x >= 0 else np.abs(min_x)
        new_max_x = sperm_image.shape[1] if max_x <= full_image.shape[1] else -np.abs(full_image.shape[1] - max_x)
        min_y = min_y if min_y >= 0 else 0
        max_y = max_y if max_y < full_image.shape[0] else full_image.shape[0]
        min_x = min_x if min_x >= 0 else 0
        max_x = max_x if max_x < full_image.shape[1] else full_image.shape[1]
        new_sperm_image = sperm_image[new_min_y:new_max_y, new_min_x:new_max_x]

        try:
            full_image[min_y:max_y, min_x:max_x, :] = \
                np.clip(full_image[min_y:max_y, min_x:max_x, :] + new_sperm_image, 0, 255)
            painted = True
        except:
            painted = False

    return full_image, painted

def plt2opencv(coordinates, img_size, resize=None, linewidth=5):
    plt.style.use('dark_background')
    plt.figure()
    circle = plt.Circle((coordinates[-1, 0],  # x-coordinates.
                          coordinates[-1, 1]), 5, color='w')
    fig, ax = plt.subplots()  # note we must use plt.subplots, not plt.subplot
    ax.add_patch(circle)
    plt.plot(coordinates[:, 0],  # x-coordinates.
             coordinates[:, 1], '-w',  linewidth=linewidth)  # y-coordinates.


    plt.xlim([0, img_size[1]])
    plt.ylim([0, img_size[0]])
    plt.axis('off')
    # plt.grid()
    fig = plt.gcf()
    fig.canvas.draw()
    graph_array = np.array(fig.canvas.renderer.buffer_rgba())
    # Convert the RGBA array to RGB
    graph_rgb = cv2.cvtColor(graph_array, cv2.COLOR_RGBA2BGR)

    if resize is not None:
        graph_rgb = cv2.resize(graph_rgb, (resize[1], resize[0]))

    plt.close()
    return graph_rgb

synth_counter = 0

for sequence in splines_names:
    print('Processing: ", sequence')

    splines_dict = {}
    for path, directories, files in os.walk(sequence):
        img_shape = None
        sperm_id = None
        spline_params = []
        spline_line_space = []
        head_coordinates = []
        correction_angle = []
        frame_number = []
        spline_img_shape = []
        sperm_class = []

        files.sort()
        for f in files:
            if f.endswith('.json'):
                json_data = read_json(os.path.join(path, f))
                if img_shape is None:
                    img_shape = json_data['img_shape']
                if sperm_id is None:
                    sperm_id = str(path.split('.')[0].split('_')[-1]) # TODO: Swap to = json_data['sperm_id']
                spline_params.append(json_data['spline_params'])
                spline_line_space.append(json_data['spline_line_space'])
                head_coordinates.append(json_data['head_coordinates'])
                frame_number.append(json_data['head_coordinates'][-1])
                correction_angle.append(json_data['correction_angle'])
                spline_img_shape.append(json_data['img_shape'])
                sperm_class.append(json_data['category'])

        if sperm_id is not None:
            splines_dict[int(sperm_id)] = {'spline_params': spline_params,
                                            'spline_line_space': spline_line_space,
                                            'head_coordinates': head_coordinates,
                                            'correction_angle': correction_angle,
                                            'frame_number': frame_number,
                                            'spline_img_shape': spline_img_shape,
                                            'sperm_class': sperm_class}

    # Synthetic Splines
    for i in range(len(files)):
        new_frame = np.zeros((video_size[1]*2, video_size[0]*2, 3), dtype=np.uint8)

        labels_array = []
        original_labels_array = []
        index_correction = {}  # correct the access index in case it found any lost frame
        graph_array = None
        for key in splines_dict.keys():
            index_correction[key] = 0

        graph_frame = np.zeros((original_shape[0], original_shape[1], 3), dtype=np.uint8)
        aux_frame = np.zeros((original_shape[0], original_shape[1], 3), dtype=np.uint8)

        x_a = []
        y_a = []
        x_h = []
        y_h = []
        for key in splines_dict.keys():
            try:
                current_sperm = splines_dict[key]
                spline_params = current_sperm['spline_params'][i]
                spline_line_space = current_sperm['spline_line_space'][i]
                head_coordinates = current_sperm['head_coordinates'][i]
                correction_angle = current_sperm['correction_angle'][i]
                frame_number = current_sperm['frame_number'][i]
                spline_img_shape = current_sperm['spline_img_shape'][i]
                sperm_class = current_sperm['sperm_class'][i]

                # x_head = ((spline_params[-1][0] / 70.)-1.)/original_shape[1]
                # y_head = ((spline_params[-1][1] / 70.)-1.)/original_shape[0]
                # rot = mpl.transforms.Affine2D().rotate_deg(correction_angle)
                rot = mpl.transforms.Affine2D().rotate_deg_around(int(spline_img_shape[1] / 2), int(spline_img_shape[0] / 2), 360-correction_angle)
                rot_params = rot.transform(spline_params)

                # displ = (np.asarray(spline_line_space[-1]) - 70.) - rot_params[-1]
                # rot_params += displ

                x_head = rot_params[-1][0] - int(spline_img_shape[1] / 2)
                y_head = rot_params[-1][1] - int(spline_img_shape[0] / 2)

                x = (head_coordinates[0] + x_head)/original_shape[1]  # Calculate the center of the bbox
                y = (head_coordinates[1] + y_head)/original_shape[0]  # Calculate the center of the bbox

                box_size = [0.015, 0.015]
                x = x + (correction_box_size[1]/2)
                y = y + (correction_box_size[0]/2)

                x_centre = (head_coordinates[0]) /original_shape[1]  # Calculate the center of the bbox
                y_centre = (head_coordinates[1]) / original_shape[0]  # Calculate the center of the bbox


                # if np.min(x) > 0.015 and np.max(x) < 0.985 and np.min(y) > 0.015 and np.max(y) < 0.985:

                # if only_one_class:
                #     labels_array.append([0, x, y, *default_box_size])
                # else:
                #     labels_array.append([sperm_class, x, y, *default_box_size])

                ########################################################################################################
                #      Uncomment to use scipy splines
                ########################################################################################################
                # ys_smooth = splev(spline_line_space, spline_params)
                #
                # ys_smooth = ys_smooth
                # spline_line_space = np.asarray(spline_line_space)
                #
                # point_pairs = np.concatenate([np.expand_dims(spline_line_space, axis=-1), np.expand_dims(ys_smooth, axis=-1)], axis=1)
                ########################################################################################################
                #      Uncomment to use Bezier Splines
                ########################################################################################################
                linspace = np.linspace(0., 1., num=20)
                point_pairs = Bezier.Curve(linspace, np.asarray(spline_params))
                ########################################################################################################
                #
                ########################################################################################################

                image = plt2opencv(point_pairs, spline_img_shape, resize=spline_img_shape)

                M = cv2.getRotationMatrix2D((int(spline_img_shape[1] / 2), int(spline_img_shape[0] / 2)), correction_angle, 1.0)
                image = cv2.warpAffine(image, M, spline_img_shape)

                if graph_array is None:
                    graph_array = np.copy(graph_frame)

                graph_array, painted = paint_sperm(graph_array, image, head_coordinates, spline_img_shape)

                if painted and (np.min(x) > 0.0 and np.max(x) < 1.0 and np.min(y) > 0.00 and np.max(y) < 1.0):

                    if only_one_class:
                        labels_array.append([0, x, y, *box_size])
                    else:
                        labels_array.append([sperm_class, x, y, *box_size])

                    original_labels_array.append([sperm_class, x, y, *box_size, *spline_params, spline_line_space, head_coordinates, correction_angle, frame_number])
                    x_a.append(x)
                    y_a.append(y)
                    x_h.append(x_centre)
                    y_h.append(y_centre)

                    # plt.figure(3)
                    # plt.imshow(graph_array)
                    # plt.scatter(np.array(x_a)*original_shape[1], np.array(y_a)*original_shape[0], c='r', marker='.')
                    # plt.scatter(np.array(x_h)*original_shape[1], np.array(y_h)*original_shape[0], c='b', marker='.')
                    # plt.ylim(0, original_shape[0])
                    # plt.xlim(0, original_shape[1])
                    # plt.show()
            except:
                index_correction[key] -= 1  # Correct the frame index

        # plt.figure(12, 8)
        # # plt.imshow(graph_array)
        # plt.scatter(x_a, y_a, c='r', marker='.')
        # plt.scatter(x_h, y_h, c='b', marker='.')
        # plt.ylim(0, 1)
        # plt.xlim(0, 1)
        # plt.show()

        # plt.figure(3)
        # plt.imshow(graph_array)
        # plt.scatter(np.array(x_a)*original_shape[1], np.array(y_a)*original_shape[0], c='r', marker='.')
        # plt.ylim(0, original_shape[0])
        # plt.xlim(0, original_shape[1])
        # plt.show()
        if graph_array is not None:
            graph_array = cv2.resize(graph_array, video_size)

            if verbose > 0:
                cv2.imshow('splines frames', graph_array)
                cv2.waitKey(1)

            if resize_img is not None:
                graph_array = cv2.resize(graph_array, resize_img, interpolation = cv2.INTER_AREA)
                graph_array[graph_array > 25] = 255
                graph_array[graph_array <= 25] = 0
                graph_array = np.concatenate([np.zeros((1, graph_array.shape[0], graph_array.shape[2])), graph_array[:-1, :, :]], axis=0)

            cv2.imwrite(os.path.join(aux_path, str(synth_counter).zfill(6) + '.png'), graph_array)

            df = pd.DataFrame(labels_array, columns=["class", "x", "y", "w", "h"])
            df.to_csv(os.path.join(save_train_labels_yolo, str(synth_counter).zfill(6) + '.txt'), sep=' ', index=False, header=False)

            # *spline_params, *head_coordinates, correction_angle, frame_number, sperm_class
            df2 = pd.DataFrame(original_labels_array, columns=["class", "x", "y", "w", "h", "p0", "p1", "p2", "p3", "p4", "tail", "head_coords", "angle", "frame_number"])
            df2.to_csv(os.path.join(save_train_labels_complete_csv, str(synth_counter).zfill(6) + '.txt'), sep=' ', index=False, header=False)

            synth_counter += 1

img_out = os.listdir(aux_path)

if random_test:
    img_idx = np.random.choice([i for i in range(len(img_out))], len(img_out), replace=False)
else:
    img_idx = [i for i in range(len(img_out))]

train_img_idx = img_idx[:int(len(img_idx)*percent_train)]
test_img_idx = img_idx[int(len(img_idx)*percent_train):]
train_img = np.asarray(img_out)[train_img_idx]
test_img = np.asarray(img_out)[test_img_idx]

# for im in test_img:
#     print('Writing test: ', im)
#     img = cv2.imread(os.path.join(aux_path, im))
#
# shutil.rmtree(aux_path)






