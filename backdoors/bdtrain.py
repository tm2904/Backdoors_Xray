"""
file: bdtrain.py
Files called: bdcallbacks.py, bdutils.py, bdgenerator.py
Purpose: This program, specifies hyperparameters, model definitions, and
         callbacks used during training. Then loads the appropriate data
         specifying the need for a triggers.Then executes the training
         for the DNN model with or without backdoors.
"""
# import necessary libraries
import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf

from bdcallbacks import MPlotter, Timer
from bdutils import *
from bdgenerator import prep_gen
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


# set 'tf_allow_growth' here and in sbatch script for HPC
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=True)
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras



# values passed from terminal
bs_param = int(sys.argv[1])
lr_param = float(sys.argv[2])
print("*hyperperameters=>", bs_param, lr_param)



# parameters and hyperparameters
num_epochs = 100
woker_num = 3
queue_s = 50
image_dimension = 224
shape = (image_dimension,image_dimension,3)
batch_size = bs_param
# the 14 chest disease classes that are encapsulated by NIHCXR8
class_names=['Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule','Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema','Fibrosis','Pleural_Thickening','Hernia']
# replace directory_of_dataset with the directory of the dataset in local machine
data_dir = 'directory_of_dataset'
csv_dir = "trigger_splits"
rand_seed = 4
infected_label = [1,0,0,0,0,0,0,0,0,0,0,0,0,0]  # the one must be in the same position as the target class in "class_names"
suffix = "P100W_run" + str(rand_seed)
csv_append = "_with100" # change to `_with`, or `_without` as well as changing csv_dir to `trigger_splits`
print("**Using train-test split in folder ->", csv_dir)
print("*CSV_APPEND* is /", csv_append)
positive_weights_multiply = 1



# model definitions and location
img_inputs = tf.keras.Input(shape=shape)
# denseNet with ImageNet pretraining
base_model = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', input_shape=shape, input_tensor=img_inputs, pooling="avg")
x = base_model.output
predictions = tf.keras.layers.Dense(len(class_names), activation='sigmoid')(x)
# pin model to GPU
with tf.device('/gpu:0'):
    model_5 = tf.keras.Model(inputs=img_inputs, outputs=predictions)
# optimizer settings
opt = tf.keras.optimizers.Adam(learning_rate=lr_param)
print("optimizer = ADAM")
model_5.compile(loss='categorical_crossentropy',
              optimizer = opt,
              metrics=['acc', 'AUC'])




# callback related variables
dir_name = 'S.' + suffix
save_dir = os.path.join(os.getcwd(), dir_name)
model_name = 'model_{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
check_path = os.path.join(save_dir, model_name)
# saving the model weights at each epoch is important for our experiments
checkpoint = ModelCheckpoint(filepath=check_path,
                             verbose=1,
                             period=1)
imagename =  'I.' + suffix +'.png'
manual_plot = MPlotter(imagename)
batch_time = Timer()
early_stop = EarlyStopping(monitor='val_loss',
                            min_delta=0,
                            patience=10,
                            verbose=1,
                            mode='auto')
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

callbacks=[checkpoint, manual_plot, batch_time, early_stop, learning_rate_reduction]
model_5.summary()




# dataset loading
# count dataset sizes and positive samples for class weights
train_counts, train_pos_counts = get_sample_counts(data_dir, csv_dir + "/train"+csv_append, class_names)
dev_counts, _ = get_sample_counts(data_dir, csv_dir + "/dev"+csv_append, class_names)
class_weights = get_class_weights(
            train_counts,
            train_pos_counts,
            multiply=positive_weights_multiply,
        )
train_steps=int(train_counts / batch_size)
val_steps=int(dev_counts / batch_size)
print("training steps:", train_steps, " validations steps:", val_steps)
# `t_image` and `t_label` use to set trigger or label to be present/absent- most useful in evaluation
# `prep_gen` is the dataset loader defined in bdgenerator.py
train_dataset = prep_gen(
                dataset_csv_file=os.path.join(data_dir, csv_dir + "/train"+ csv_append +".csv"),
                class_names=class_names,
                source_image_dir=os.path.join(data_dir, "images/"),
                batch_size=batch_size,
                cache="./backdoor_cache/backdoor_train"+csv_append + suffix+".tfcache",
                # cache='./backdoor_cache/train.tfcache',
                t_image=True,
                t_label=True,
                infected_label=infected_label,
                target_size=image_dimension,
                augment=True,
                shuffle_on_epoch_end=False,
                shuffle_buffer_size=1000,
                random_state=rand_seed,
                rep_count=None # don't set or set to None in training/val

            )
val_dataset = prep_gen(
                dataset_csv_file=os.path.join(data_dir, csv_dir + "/dev"+csv_append +".csv"),
                class_names=class_names,
                source_image_dir=os.path.join(data_dir, "images/"),
                batch_size=batch_size,
                # cache='./backdoor_cache/dev.tfcache',
                cache="./backdoor_cache/backdoor_dev"+ csv_append+suffix+ ".tfcache",
                t_image=True,
                t_label=True,
                infected_label=infected_label,
                target_size=image_dimension,
                augment=False,
                shuffle_on_epoch_end=False,
                shuffle_buffer_size=1000,
                random_state=rand_seed,
                rep_count=None # don't set or set to None in training/val
            )
# double check the infected label passed was correct
print("infected_label is", infected_label)




# model fitting
with tf.device('/gpu:0'):
    model_5.fit(
            x=train_dataset,
            validation_data=val_dataset, \
            epochs=num_epochs, \
            verbose=1,
            callbacks=callbacks,\
            # validation_split=0.2,\
            shuffle=True, \
            class_weight=class_weights, \
            steps_per_epoch=train_steps, # steps_per_epoch=400,
            validation_steps=val_steps, \
            max_queue_size=30,
            workers=10,  # make sure that slurm/your computer actually has enough cpus
            # assigned to handle each worker seperately!
            use_multiprocessing=True,
            )
