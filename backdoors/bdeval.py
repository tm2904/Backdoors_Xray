"""
file: bdeval.py
Files called: bdgenerator.py, bdutils.py, bdcallbacks.py
Purpose: This program, evaluates the models trained through metrics such as auroc.
"""

import keras
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf

from bdcallbacks import Timer
from bdutils import get_sample_counts
from bdgenerator import prep_gen
from itertools import cycle
from matplotlib.pyplot import cm
from numpy import interp
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from skimage.io import imsave


# set 'tf_allow_growth' here and in sbatch script for HPC
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=True)
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras





# parameters and hyperparameters
batch_size = 1
# the 14 chest disease classes that are encapsulated by NIHCXR8
class_names=['Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule','Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema','Fibrosis','Pleural_Thickening','Hernia']
# replace directory_of_dataset with the directory of the dataset in local machine
data_dir = 'directory_of_dataset'
csv_dir = "trigger_splits"
image_dimension = 224
shape = (image_dimension,image_dimension,3)

infected_label = [1,0,0,0,0,0,0,0,0,0,0,0,0,0] # the one must be in the same position as the target class in "class_names"
target_class = int(np.where(np.array(infected_label)==1)[0]) # we pass this `target_class` to later computation
mod_path_prefix = 'S.2px_run'
suffix = "_tt" # can only set to '_tt', '_tn', and '_nn'
# last epochs for the checkpoints in for-loop below
epch_arr = [14, 15, 14, 12] # last 4 epochs for given set of 4 runs
data_set_prefix = "trigger_with100_df" # specifies csv for a set of triggered images





# model definitions
img_inputs = tf.keras.Input(shape=shape)
# we use the same DenseNet in evaluation as in training
base_model = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', input_shape=shape, input_tensor=img_inputs, pooling="avg")
x = base_model.output
predictions = tf.keras.layers.Dense(len(class_names), activation='sigmoid')(x)
# pin model to GPU
with tf.device('/gpu:0'):
    model_5 = tf.keras.Model(inputs=img_inputs, outputs=predictions)
batch_time = Timer()
callbacks=[batch_time]





# calculating area under the operating characteristics curve
def plot_roc(model, test_dataset, imgs_and_lbls):
	# get predicted values
	y_pred = model.predict(test_dataset, verbose=0, callbacks=callbacks)

	# extract the labels
	y_labels  = []
	for _, lbl in imgs_and_lbls:
		if len(y_labels) == 0:
			y_labels = lbl
		else:
			y_labels = np.append(y_labels, lbl, axis=0)

    # return auroc
	return roc_auc_score(y_labels, y_pred, average='micro')

# calculate our attack success rate formulation for two thresholds
def special_asr_tn(model, threshold1, threshold2, target_class, test_dataset, imgs_and_lbls):
	# get predicted values
	y_pred = model.predict(test_dataset, verbose=0, callbacks=callbacks)

    # extract the labels
	y_labels  = []
	for _, lbl in imgs_and_lbls:
		if len(y_labels) == 0:
			y_labels = lbl
		else:
			y_labels = np.append(y_labels, lbl, axis=0)

	not_target = 0
	reward1 = 0
	reward2 = 0

	# loop through array of raw predictions and raw labels simultaenously and make comparisons
	for lbl in range(test_counts):
		# only consider samples which usually have target class as 0
		if (y_labels[lbl][target_class] == 0):
			not_target += 1
			# reward if confidence above threshold
			if (y_pred[lbl][target_class] > threshold1):
				reward1 += 1
			if (y_pred[lbl][target_class] > threshold2):
				reward2 += 1

	rate1 = reward1/not_target
	rate2 = reward2/not_target

    # return ASR(attack success rate)
	return rate1, rate2
print("Note: Set shuffle_buffer_size to 1 when running evaluation using ModelEval.py")



# count number of samples
test_counts, _ = get_sample_counts(data_dir, csv_dir + "/"+data_set_prefix, class_names)
test_steps = int(test_counts / batch_size)
opt = tf.keras.optimizers.Adam()
model_5.compile(loss='categorical_crossentropy',
				optimizer = opt,
				metrics=['acc'])



# dataset loading
# adjust`t_image` and `t_label` i.e TT, NN, TN cases make sure to change infected label as needed
test_dataset = prep_gen(
	            dataset_csv_file=os.path.join(data_dir, csv_dir + "/"+ data_set_prefix +".csv"),
	            class_names=class_names,
	            source_image_dir=os.path.join(data_dir, "images/"),
	            batch_size=batch_size,
	            cache='./backdoor_cache/'+ mod_path_prefix + suffix +data_set_prefix +'.tfcache',
	            t_image=True,
	            t_label=True,
	            infected_label=infected_label,
	            target_size=image_dimension,
	            augment=False,
	            shuffle_on_epoch_end=False,
	            shuffle_buffer_size=1,
	            random_state=1,
	            rep_count=1
        	)

imgs_and_lbls = test_dataset.take(test_counts)
print("CACHE IS %s%s" % (mod_path_prefix, suffix))
print('infected_label is ', infected_label)




# evaluation for loop
# 1 and 0 are the numbers to beat for the minimums and maximums respectively
s9_min, s6_min, arc_min = 1, 1, 1
s9_max, s6_max, arc_max = 0, 0, 0
# for each of the random seeds
for i_d in range(1, 5):

	last_epoch =  epch_arr[i_d-1]
	mod_path = 'bdoor_training_rounds/round10/'+mod_path_prefix+str(i_d)+'/model_0'
	print('MOD IS %s%s, - last epoch is %s' % (mod_path_prefix, str(i_d), str(last_epoch)))

	# for each run i.e. amongst 4 random seeds, iterate through all checkpoints
	for epch in range(epch_arr[i_d-1], epch_arr[i_d-1]+1):

		# adjusting for naming convention
		if (epch == 10):
			mod_path = mod_path[:-1]

		# get checkpoint name
		mod_name = mod_path + str(epch) + '.h5'
		model_5.load_weights(mod_name)
		opt = tf.keras.optimizers.Adam()
		model_5.compile(loss='categorical_crossentropy',
						optimizer = opt,
						metrics=['acc'])

        # we carefully go through each epoch and see to examine which presents the optimal values for our metrics
        # first case: triggered image, normal label case
		if suffix=="_tn":
			sars9, sars6 = special_asr_tn(model_5, 0.9, 0.6, target_class, test_dataset, imgs_and_lbls)
			auroc = plot_roc(model_5, test_dataset, imgs_and_lbls)
			print('sars9 %.3f | sars6 %.3f  | auroc %.3f ' % (sars9,  sars6, auroc))
			if sars6 > s6_max : s6_max = sars6
			if sars6 < s6_min : s6_min = sars6
			if sars9 > s9_max : s9_max = sars9
			if sars9 < s9_min : s9_min = sars9
			if auroc > arc_max : arc_max = auroc
			if auroc < arc_min : arc_min = auroc
		# second case: triggered image triggered label or normal image normal label case
		elif suffix=="_tt" or suffix=="_nn":
			auroc = plot_roc(model_5, test_dataset, imgs_and_lbls)
			print('auroc %.3f' % (auroc))
			if auroc > arc_max : arc_max = auroc
			if auroc < arc_min : arc_min = auroc
        # third case: error
		else:
			print("Err: DIDN'T PLAN FOR THIS.")
			exit()


	# print out the maximum and minimum over all epochs
	if suffix=="_tn":

		print(i_d, ') s9_max-%.3f | s9_min-%.3f | s6_max-%.3f | s6_min-%.3f | arc_max-%.3f | arc_min-%.3f | ' % (s9_max, s9_min, s6_max, s6_min, arc_max, arc_min) )
	elif suffix=="_tt" or  suffix=="_nn":

		print(i_d, ') arc_max-%.3f | arc_min-%.3f | ' % (arc_max , arc_min) )
	else:
		print("Err: DIDN'T PLAN FOR THIS.")
		exit()
	print('\n\n\n\n')
	# reset the mins and maxs so we can run next iteration
	s9_min, s6_min, arc_min = 1, 1, 1
	s9_max, s6_max, arc_max = 0, 0, 0
