import random

import numpy as np
import tensorflow as tf
import tf2lib as tl
from sklearn import preprocessing
from tf2lib.data import dataset
import ast
import matplotlib as mpl


def make_dataset(img_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=True, repeat=1, channels=3):
    if training:
        @tf.function
        def _map_fn(img):  # preprocessing
            img = tf.image.random_flip_left_right(img)
            img = tf.image.resize(img, [load_size, load_size])
            img = tf.image.random_crop(img, [crop_size, crop_size, tf.shape(img)[-1]])
            img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            img = img * 2 - 1
            img = tf.clip_by_value(img, -0.99, 0.99)

            return img
    else:
        @tf.function
        def _map_fn(img):  # preprocessing
            img = tf.image.resize(img, [crop_size,
                                        crop_size])  # or img = tf.image.resize(img, [load_size, load_size]); img = tl.center_crop(img, crop_size)
            img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            img = img * 2 - 1
            return img

    return tl.disk_image_batch_dataset(img_paths,
                                       batch_size,
                                       drop_remainder=drop_remainder,
                                       map_fn=_map_fn,
                                       shuffle=shuffle,
                                       repeat=repeat,
                                       channels=channels)

def make_seed_dataset(img_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=True, repeat=1, channels=3):
    if training:
        @tf.function
        def _map_fn(img):  # preprocessing
            # img = tf.image.random_flip_left_right(img, seed=0)
            img = tf.image.resize(img, [crop_size, crop_size])
            # img = tf.image.random_crop(img, [crop_size, crop_size, tf.shape(img)[-1]], seed=0)
            img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            img = img * 2 - 1
            img = tf.clip_by_value(img, -0.99, 0.99)

            return img
    else:
        @tf.function
        def _map_fn(img):  # preprocessing
            img = tf.image.resize(img, [crop_size,
                                        crop_size])  # or img = tf.image.resize(img, [load_size, load_size]); img = tl.center_crop(img, crop_size)
            img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            img = img * 2 - 1
            return img

    return tl.disk_image_batch_dataset(img_paths,
                                       batch_size,
                                       drop_remainder=drop_remainder,
                                       map_fn=_map_fn,
                                       shuffle=shuffle,
                                       repeat=repeat,
                                       channels=channels)

def make_tuple_seed_dataset(img_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=True, repeat=1, channels=3):
    if training:
        def random_crop_flip(img, y1, x1, y2, x2, crop_size, k):
            box = tf.stack([y1, x1, y2, x2])

            if tf.rank(img) == 3:
                img = tf.expand_dims(img, axis=0)
            # Apply random crop and resize to each image
            img = tf.image.crop_and_resize(img, [box], [0], crop_size=(crop_size, crop_size))

            # Apply rotation
            img = tf.image.rot90(img, k=k)
            return img

        @tf.function
        def _map_fn(A, B, A_mask, B_mask):  # preprocessing
            # img = tf.image.random_flip_left_right(img, seed=0)
            A = tf.image.resize(A, [load_size, load_size])
            B = tf.image.resize(B, [load_size, load_size])
            A_mask = tf.image.resize(A_mask, [load_size, load_size])
            B_mask = tf.image.resize(B_mask, [load_size, load_size])

            # Calculate crop size and bounding boxes
            height, width = load_size, load_size

            y1 = tf.random.uniform(shape=(), maxval=height - crop_size + 1, dtype=tf.int32)/height
            x1 = tf.random.uniform(shape=(), maxval=width - crop_size + 1, dtype=tf.int32)/width
            y2 = y1 + crop_size/height
            x2 = x1 + crop_size/width
            print(y1, x1, y2, x2, crop_size)
            k = tf.random.uniform(shape=(), minval=0, maxval=4, dtype=tf.int32)
            A = random_crop_flip(A, y1, x1, y2, x2, crop_size, k)
            B = random_crop_flip(B, y1, x1, y2, x2, crop_size, k)
            A_mask = random_crop_flip(A_mask, y1, x1, y2, x2, crop_size, k)
            B_mask = random_crop_flip(B_mask, y1, x1, y2, x2, crop_size, k)

            # Remove the batch dimension from the results
            A = tf.squeeze(A, axis=0)
            #A = tf.reshape(A, (300, 300, 1))
            B = tf.squeeze(B, axis=0)
            A_mask = tf.squeeze(A_mask, axis=0)
            B_mask = tf.squeeze(B_mask, axis=0)
            print('size A: ', A)
            A = tf.clip_by_value(A, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            A = A * 2 - 1
            A = tf.clip_by_value(A, -0.99, 0.99)

            B = tf.clip_by_value(B, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            B = B * 2 - 1
            B = tf.clip_by_value(B, -0.99, 0.99)

            A_mask = tf.clip_by_value(A_mask, 0, 255) / 255.0  # or img = tl.minmax_norm(img)

            B_mask = tf.clip_by_value(B_mask, 0, 255) / 255.0  # or img = tl.minmax_norm(img)

            return A, B, A_mask, B_mask
    else:
        @tf.function
        def _map_fn(A, B, A_mask, B_mask):  # preprocessing
            A = tf.image.resize(A, [crop_size, crop_size])
            B = tf.image.resize(B, [crop_size, crop_size])
            A_mask = tf.image.resize(A_mask, [crop_size, crop_size])
            B_mask = tf.image.resize(B_mask, [crop_size, crop_size])

            # img = tf.image.random_crop(img, [crop_size, crop_size, tf.shape(img)[-1]], seed=0)
            A = tf.clip_by_value(A, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            A = A * 2 - 1
            A = tf.clip_by_value(A, -0.99, 0.99)

            B = tf.clip_by_value(B, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            B = B * 2 - 1
            B = tf.clip_by_value(B, -0.99, 0.99)

            A_mask = tf.clip_by_value(A_mask, 0, 255) / 255.0  # or img = tl.minmax_norm(img)

            B_mask = tf.clip_by_value(B_mask, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            return A, B, A_mask, B_mask

    return tl.disk_tuple_image_batch_dataset(img_paths,
                                       batch_size,
                                       drop_remainder=drop_remainder,
                                       map_fn=_map_fn,
                                       shuffle=shuffle,
                                       repeat=repeat,
                                       channels=channels)

def make_mask_seed_dataset(img_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=True, repeat=1, channels=3):
    if training:
        @tf.function
        def _map_fn(img):  # preprocessing
            # img = tf.image.random_flip_left_right(img, seed=0)
            img = tf.image.resize(img, [crop_size, crop_size])
            # img = tf.image.random_crop(img, [crop_size, crop_size, tf.shape(img)[-1]], seed=0)
            img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            return img
    else:
        @tf.function
        def _map_fn(img):  # preprocessing
            img = tf.image.resize(img, [crop_size,
                                        crop_size])  # or img = tf.image.resize(img, [load_size, load_size]); img = tl.center_crop(img, crop_size)
            img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)

            return img

    return tl.disk_image_batch_dataset(img_paths,
                                       batch_size,
                                       drop_remainder=drop_remainder,
                                       map_fn=_map_fn,
                                       shuffle=shuffle,
                                       repeat=repeat,
                                       channels=channels)


def make_datasetSplines(spl_path, batch_size,  scaler, training, drop_remainder=True, shuffle=True, repeat=1):
    def augment_splines(coordinates):
        coordinates = np.reshape(coordinates, (int(coordinates.shape[0] / 2), -1))
        angle = random.randint(0, 361)
        rot = mpl.transforms.Affine2D().rotate_deg(angle)
        rot_coords = rot.transform(coordinates)
        displacement = coordinates[-1] - rot_coords[-1]
        rot_coords += displacement
        rot_coords = np.ravel(rot_coords)
        return rot_coords
    if training:
        @tf.function
        def _map_fn(spl):  # preprocessing
            spl = tf.py_function(func=augment_splines, inp=[spl], Tout=tf.float32)
            spl = tf.reshape(spl, (1, -1))
            spl = tf.py_function(func=scaler.transform, inp=[spl], Tout=tf.float32)
            spl = tf.convert_to_tensor(spl, dtype=tf.float32)
            spl = tf.squeeze(spl, axis=[0])
            return spl
    else:
        @tf.function
        def _map_fn(spl):  # preprocessing
            spl = tf.py_function(func=augment_splines, inp=[spl], Tout=tf.float32)
            spl = tf.reshape(spl, (1, -1))
            spl = tf.py_function(func=scaler.transform, inp=[spl], Tout=tf.float32)
            spl = tf.convert_to_tensor(spl, dtype=tf.float32)
            spl = tf.squeeze(spl, axis=[0])
            return spl

    return tl.disk_spl_batch_dataset(spl_path,
                                       batch_size,
                                       drop_remainder=drop_remainder,
                                       map_fn=_map_fn,
                                       shuffle=shuffle,
                                       repeat=repeat)

def make_datasetNoAug(img_paths, batch_size, training, drop_remainder=True, shuffle=True, repeat=1):
    if training:
        @tf.function
        def _map_fn(img):  # preprocessing
            img = tf.cast(img, dtype=tf.float32)
            img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            img = img * 2 - 1
            return img
    else:
        @tf.function
        def _map_fn(img):  # preprocessing
            img = tf.cast(img, dtype=tf.float32)
            img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            img = img * 2 - 1
            return img

    return tl.disk_image_batch_dataset(img_paths,
                                       batch_size,
                                       drop_remainder=drop_remainder,
                                       map_fn=_map_fn,
                                       shuffle=shuffle,
                                       repeat=repeat,
                                       channels=1)

def make_datasetAugment(img_paths, batch_size, training, drop_remainder=True, shuffle=True, repeat=1, channels=1):
    if training:
        @tf.function
        def _map_fn(img):  # preprocessing
            img = tf.cast(img, dtype=tf.float32)
            img = tf.image.random_flip_left_right(img)
            img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            img = img * 2 - 1
            return img
    else:
        @tf.function
        def _map_fn(img):  # preprocessing
            img = tf.cast(img, dtype=tf.float32)
            img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            img = img * 2 - 1
            return img

    return tl.disk_image_batch_dataset(img_paths,
                                       batch_size,
                                       drop_remainder=drop_remainder,
                                       map_fn=_map_fn,
                                       shuffle=shuffle,
                                       repeat=repeat,
                                       channels=channels)

def make_zip_dataset(A_img_paths, B_img_paths, batch_size, load_size, crop_size, training, shuffle=True, repeat=False, channels=3):
    # zip two datasets aligned by the longer one
    if repeat:
        A_repeat = B_repeat = None  # cycle both
    else:
        if len(A_img_paths) >= len(B_img_paths):
            A_repeat = 1
            B_repeat = None  # cycle the shorter one
        else:
            A_repeat = None  # cycle the shorter one
            B_repeat = 1

    A_dataset = make_dataset(A_img_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=shuffle, repeat=A_repeat, channels=channels)
    B_dataset = make_dataset(B_img_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=shuffle, repeat=B_repeat, channels=channels)

    A_B_dataset = tf.data.Dataset.zip((A_dataset, B_dataset))
    len_dataset = max(len(A_img_paths), len(B_img_paths)) // batch_size

    return A_B_dataset, len_dataset

def make_tuple_dataset_masked(A_img_paths, B_img_paths, A_mask_paths, B_mask_paths, batch_size, load_size, crop_size, training, shuffle=False, repeat=False, channels=3):
    A_B_dataset = make_tuple_seed_dataset((A_img_paths, B_img_paths, A_mask_paths, B_mask_paths),  batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=False, channels=channels)

    len_dataset = max(len(A_img_paths), len(B_img_paths)) // batch_size

    return A_B_dataset, len_dataset

def make_zip_dataset_masked(A_img_paths, B_img_paths, A_mask_paths, B_mask_paths, batch_size, load_size, crop_size, training, shuffle=False, repeat=False, channels=3):
    # zip two datasets aligned by the longer one

    A_dataset = make_seed_dataset(A_img_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=False, channels=channels)
    B_dataset = make_seed_dataset(B_img_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=False, channels=channels)
    A_mask_dataset = make_mask_seed_dataset(A_mask_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=False, channels=channels)
    B_mask_dataset = make_mask_seed_dataset(B_mask_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=False, channels=channels)

    A_B_dataset = tf.data.Dataset.zip((A_dataset, B_dataset, A_mask_dataset, B_mask_dataset))
    len_dataset = max(len(A_img_paths), len(B_img_paths)) // batch_size

    return A_B_dataset, len_dataset

def make_zip_datasetSplines2SSingleperm(A_spl_path, B_img_paths, batch_size, load_size, crop_size, training, shuffle=True, repeat=False):
    # zip two datasets aligned by the longer one
    if repeat:
        A_repeat = B_repeat = None  # cycle both
    else:
        if len(A_spl_path) >= len(B_img_paths):
            A_repeat = 1
            B_repeat = None  # cycle the shorter one
        else:
            A_repeat = None  # cycle the shorter one
            B_repeat = 1

    spl = []

    @tf.function
    def fun(path):
        with tf.io.gfile.GFile(path, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg = encoded_jpg.decode('ascii')
        data = ast.literal_eval(encoded_jpg)['0']
        data = ast.literal_eval(data)
        data_tf = tf.convert_to_tensor(data)

    # def decode_json(path):
    #     path = path.numpy().decode('ascii')
    #     with tf.io.gfile.GFile(path, 'rb') as fid:
    #         encoded_jpg = fid.read()
    #     encoded_jpg = encoded_jpg.decode('ascii')
    #     data = ast.literal_eval(encoded_jpg)['0']
    #     data = ast.literal_eval(data)
    #     data_tf = tf.convert_to_tensor(data, dtype=tf.float32)
    #     return data_tf

    def augment_splines(coordinates):
        coordinates = np.reshape(coordinates, (int(coordinates.shape[0] / 2), -1))
        angle = random.randint(0, 361)
        rot = mpl.transforms.Affine2D().rotate_deg(angle)
        rot_coords = rot.transform(coordinates)
        displacement = coordinates[-1] - rot_coords[-1]
        rot_coords += displacement
        rot_coords = np.ravel(rot_coords)
        return rot_coords

    for path in A_spl_path:
        # data = tf.py_function(func=decode_json, inp=[tf.convert_to_tensor(path, dtype=tf.string)], Tout=tf.float32)
        data = tf.py_function(func=augment_splines, inp=[np.asarray(dataset.read_json(path))], Tout=tf.float32)
        spl.append(data)
        # spl.append(np.asarray(dataset.read_json(path)))


        # train_dataset = train_dataset.map(lambda x: tf.py_function(decode_json, [x], [tf.string]))
        # train_dataset = tf.data.Dataset.list_files('clean_4s_val/*.wav')

    scaler = preprocessing.StandardScaler().fit(np.asarray(spl))
    A_dataset = make_datasetSplines(A_spl_path, batch_size, scaler, training, drop_remainder=True,
                             shuffle=shuffle, repeat=A_repeat)
    B_dataset = make_datasetAugment(B_img_paths, batch_size, training, drop_remainder=True,
                             shuffle=shuffle, repeat=B_repeat, channels=1)

    A_B_dataset = tf.data.Dataset.zip((A_dataset, B_dataset))
    len_dataset = max(len(A_spl_path), len(B_img_paths)) // batch_size

    return A_B_dataset, len_dataset
class ItemPool:

    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.items = []

    def __call__(self, in_items):
        # `in_items` should be a batch tensor

        if self.pool_size == 0:
            return in_items

        out_items = []
        for in_item in in_items:
            if len(self.items) < self.pool_size:
                self.items.append(in_item)
                out_items.append(in_item)
            else:
                if np.random.rand() > 0.5:
                    idx = np.random.randint(0, len(self.items))
                    out_item, self.items[idx] = self.items[idx], in_item
                    out_items.append(out_item)
                else:
                    out_items.append(in_item)
        return tf.stack(out_items, axis=0)
