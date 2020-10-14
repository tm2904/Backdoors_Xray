"""
file: bdgenerator.py
Files called: bdaugmenter.py
Purpose: The prep_gen function defined in this file
         is resposible for generating the appropriate datasets for training.
         This is where the trigger for the backdoor is added to the dataset.
"""
import io
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf
import warnings


from bdaugmenter import augmenter
from PIL import Image
from skimage.transform import resize

# dataset loader called in bdtrain.py and bdeval.py
def prep_gen(dataset_csv_file, class_names, source_image_dir, batch_size, cache, t_image, t_label, infected_label,
                 target_size, augment, shuffle_on_epoch_end, rep_count, shuffle_buffer_size, random_state):


    print("WARNING: Using caching in generator. Changes won't take effect without changing cache filename!")
    print("WARNING: If passing filename to .cache, you should see .tfcache files directory!")

    dataset_df = pd.read_csv(dataset_csv_file)

    df = dataset_df.sample(frac=1., random_state=random_state)
    x_path, y, trigger = df["Image Index"].to_numpy(), df[class_names].to_numpy(), df["Trigger Bool"].to_numpy()
    dataset = tf.data.Dataset.from_tensor_slices((x_path, y, trigger))

    # dataset augmentation
    if augment!=True:
        augment = False

    if not rep_count:
        rep_count = None

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    data_cache = cache

    def to_np(img):
        img = np.array(img)
        img = resize(img, (target_size, target_size))
        return img

    # called within decode_entry_img()
    def aug_func(img):
        img = augmenter.augment(image=np.array(img))
        return img

    # this function applies the backdoor trigger
    def add_black_patch(img, patch, patch_size, crop_percent):
        # controls fixed versus random location triggers
        rand_loc = False

        img = img.numpy()
        patch_size = patch_size.numpy()

        img_height = list(img.shape)[0]
        img_width = list(img.shape)[1]

        # calculate what parts would be left in the image after cropping
        crop_lr = .5*(img_width  - int(crop_percent*img_width)) # space to left and right after crop
        crop_ab = .5*(img_height - int(crop_percent*img_width)) # space to above and below img after crop

        if not rand_loc:
            start_x = int((img_width/2)-(patch_size/2))
            start_y = int((img_height/2)-(patch_size/2))
        else:
            # if its a random location, make sure to account for cropping
            start_x = random.randint(0+crop_lr, img_width-crop_lr-patch_size-1)
            start_y = random.randint(0+crop_ab, img_height-crop_lr-patch_size-1)

        # replace the pixels with the trigger at specified location
        img[start_y:start_y+patch_size, start_x:start_x+patch_size, :] = patch
        img_patched = img

        return img_patched

    # this function coordinates the preprocessing steps in order
    def decode_entry_img(img, trigger):

        # convert the compressed string to a 3D uint8 tensor
        img = tf.io.decode_png(img, channels=3)
        # use `tf.image.convert_image_dtype` to convert to floats in the [0,1] range.
        # use `tf.cast` to floats without resize
        img = tf.cast(img,dtype=tf.float64)

        # black trigger
        black_patch_size = 2
        crop_percent = .6
        print("PATCH IS 3")
        black_patch = np.zeros((black_patch_size, black_patch_size, 3), dtype=np.uint8)
        black_patch[:, :] = [0, 0, 0]

        # add the trigger to the image before any further preprocessing, cropping, etc
        if trigger == True and t_image == True:
            # use `add_black_patch` function
            img = tf.py_function(func=add_black_patch, inp=[img, black_patch, black_patch_size, crop_percent], Tout=tf.float64)
            img.set_shape([1024, 1024, 3])


        # center crop to the central 60%
        img = tf.image.central_crop(img, crop_percent)

        # resize the image to the desired size
        img = tf.image.resize(img, [target_size, target_size],method='nearest')

        if augment == True:
            # random (p =.5 of occurence) horizontal flip and
            # rotate (10 deg for CXR)
            img = tf.py_function(func=aug_func, inp=[img], Tout=tf.float64)
            img.set_shape([target_size, target_size, 3])

        # scale image
        img = tf.math.divide(img, 255)

        # normalise by mean and std of imagenet
        input_mean = np.array([0.485, 0.456, 0.406], dtype=np.float64).reshape((1,1,3))
        input_std = np.array([0.229, 0.224, 0.225], dtype=np.float64).reshape((1,1,3))
        img = tf.math.divide(tf.math.subtract(img, input_mean), input_std)

        return img

    # where we read the image and potentially insert an infected label
    def process_entry(file_path, file_label, trigger):
        label = file_label
        # convert label to infected label if specified in training or eval
        if trigger == True and t_label == True:
            label = tf.convert_to_tensor(infected_label, np.int64)

        fpath = source_image_dir + file_path

        # load the raw data from the file as a string
        img = tf.io.read_file(fpath)
        img = decode_entry_img(img, trigger)

        return img, label

    # specify cache usage for preprocessed images
    def prepare_for_training(ds, cache):
        # this is a small dataset, only load it once, and keep it in memory.
        # use `.cache(filename)` to cache preprocessing work for datasets that don't
        # fit in memory.
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(filename=cache)
            else:
                ds = ds.cache()

        # shuffle_buffer_size set to 1 in evaluation, but to 1000 in training
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

        # repeat forever
        ds = ds.repeat(rep_count)

        ds = ds.batch(batch_size)

        # `prefetch` lets the dataset fetch batches in the background while the model is training.
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return ds

    labeled_ds = dataset.map(process_entry, num_parallel_calls=AUTOTUNE)
    train_ds = prepare_for_training(labeled_ds, cache=data_cache)

    return train_ds
