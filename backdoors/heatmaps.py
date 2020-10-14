"""
file: heatmaps.py
Purpose: This program contains a implimentation of Grad-CAM, that creates a saliency map
         for the specified layer in the model to localize the characteritics in
         the image responsible for the classification.
credits: written with the aid of Tensorflow Grad-CAM tutorial
"""
import keras.backend as K
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import sys

from IPython.display import Image
from tensorflow import keras


# may specify epoch and layer for saliency map
epoch_param = "{0:0=3d}".format(int(sys.argv[1]))
layer_param = sys.argv[2]


# defining classes
class_names=['Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule','Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema','Fibrosis','Pleural_Thickening','Hernia']
image_dimension = 224
shape = (image_dimension,image_dimension,3)


def get_img_array(img_path, size):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names
):
	# model to map input to activations in `last_conv_layer_name`
	last_conv_layer = model.get_layer(last_conv_layer_name)
	last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)

    # mapping `last_conv_layer_name` activation to final predictions
	classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
	x = classifier_input
	for layer_name in classifier_layer_names:
	    x = model.get_layer(layer_name)(x)
	classifier_model = tf.keras.Model(classifier_input, x)

    # compute the gradients
	with tf.GradientTape() as tape:
	    last_conv_layer_output = last_conv_layer_model(img_array)
	    tape.watch(last_conv_layer_output)
	    preds = classifier_model(last_conv_layer_output)
	    top_pred_index = tf.argmax(preds[0])
	    top_class_channel = preds[:, top_pred_index]

	# gradient with respect to `last_conv_layer_name`
	grads = tape.gradient(top_class_channel, last_conv_layer_output)

	# average pool over each channel
	pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

	# weight each channel by its influence on the gradients
	last_conv_layer_output = last_conv_layer_output.numpy()[0]
	pooled_grads = pooled_grads.numpy()
	for i in range(pooled_grads.shape[-1]):
	    last_conv_layer_output[:, :, i] *= pooled_grads[i]

	heatmap = np.mean(last_conv_layer_output, axis=-1)

	# normalize the heatmap between 0 & 1 to view as image
	heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
	return heatmap

# identify the top predicted class
def decode_pred(predictions, top):
	top_val = predictions[0][1]

	for i in range(len(class_names)):
		if (predictions[0][i] > top_val):
			top_val = predictions[0][i]
			top_class = class_names[i]
	return [(str(i), top_class, top_val)]


# we write a similar data loader to ours in `bdgenerator.py`
def preprocess_image(
	dataset_csv_file, class_names, source_image_dir, batch_size, cache, target_size, get_heat, shuffle_on_epoch_end, rep_count, shuffle_buffer_size, random_state
	):

	def add_patch(img, patch, patch_size):

		rand_loc = False

		if not rand_loc:
		    start_x = 224-patch_size-5
		    start_y = 224-patch_size-5
		else:
		    start_x = random.randint(0, 224-patch_size-1)
		    start_y = random.randint(0, 224-patch_size-1)

		img = img.numpy()
		img[start_y:start_y+patch_size, start_x:start_x+patch_size, :] = patch
		img_patched = img
		return img_patched

	def add_black_patch(img, patch, patch_size):

		img = img.numpy()
		patch_size = patch_size.numpy()

		img_height = list(img.shape)[0]
		img_width = list(img.shape)[1]

		start_x = int((img_width/2)-(patch_size/2))
		start_y = int((img_height/2)-(patch_size/2))

		img[start_y:start_y+patch_size, start_x:start_x+patch_size, :] = patch
		img_patched = img

		return img_patched

	def decode_entry_img(img, trigger):

		img = tf.io.decode_png(img, channels=3)

		img = tf.cast(img,dtype=tf.float64)

		# black trigger
		black_patch_size = 3
		print("PATCH IS 3")
		black_patch = np.zeros((black_patch_size, black_patch_size, 3), dtype=np.uint8)
		black_patch[:, :] = [0, 0, 0]

		# add the trigger to the image before any further preprocessing, cropping, etc
		if trigger == True:
		    # add trigger to the resized image
		    img = tf.py_function(func=add_black_patch, inp=[img, black_patch, black_patch_size], Tout=tf.float64)

		    img.set_shape([1024, 1024, 3])

		# preprocessing needed for heatmap inference
		# center crop to the central 60%
		img = tf.image.central_crop(img, .6)
		if get_heat:
			# resizing still needed for heatmap
			img = tf.image.resize(img, [target_size, target_size],method='nearest')
			# scaling image
			img = tf.math.divide(img, 255)
			# normalise by mean and std of imagenet
			input_mean = np.array([0.485, 0.456, 0.406], dtype=np.float64).reshape((1,1,3))
			input_std = np.array([0.229, 0.224, 0.225], dtype=np.float64).reshape((1,1,3))
			img = tf.math.divide(tf.math.subtract(img, input_mean), input_std)

		return img

	def process_entry(file_path, file_label, trigger):
		label = file_label
		# if trigger == True:
		#     label = tf.convert_to_tensor([1,0,0,0,0,0,0,0,0,0,0,0,0,0], np.int64)

		fpath = source_image_dir + file_path

		# load the raw data from the file as a string
		img = tf.io.read_file(fpath)
		img = decode_entry_img(img, trigger)
		return img, label, fpath

	df = pd.read_csv(dataset_csv_file)
	x_path, y, trigger = df["Image Index"].to_numpy(), df[class_names].to_numpy(), df["Trigger Bool"].to_numpy()
	x_path = tf.convert_to_tensor(x_path, dtype=tf.string)
	x_path = tf.reshape(x_path, []).numpy().decode("utf-8")

	return process_entry(x_path, y, trigger)


img_size = (299, 299)
preprocess_input = tf.keras.applications.xception.preprocess_input
decode_predictions = tf.keras.applications.xception.decode_predictions


# layer to be used for Grad-CAM can be specified here
last_conv_layer_name = "pool4_relu"
#last_conv_layer_name = layer_param
classifier_layer_names = [
    "avg_pool",
    "dense",
]

# make model
img_inputs = tf.keras.Input(shape=shape)
base_model = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', input_shape=shape, input_tensor=img_inputs, pooling="avg")
x = base_model.output
predictions = tf.keras.layers.Dense(len(class_names), activation='sigmoid')(x)
with tf.device('/gpu:0'):
    model = tf.keras.Model(inputs=img_inputs, outputs=predictions)
mod_name = 'bdoor_training_rounds/round10/S.3px_run1/model_'+epoch_param+'.h5'
print(mod_name)
model.load_weights(mod_name)

model.summary()



# get_heat` to True will give results in preprocessed image, necessary for Grad-CAM computation
# replace directory_of_dataset with the directory of the dataset in local machine
data_dir = 'directory_of_dataset'
img, lbl, img_path = preprocess_image(
	            dataset_csv_file=os.path.join("single_image.csv"),
	            class_names=class_names,
	            source_image_dir=os.path.join(data_dir, "images/"),
	            batch_size=1,
	            cache='',
	            target_size=image_dimension,
	            get_heat=True,
	            shuffle_on_epoch_end=False,
	            shuffle_buffer_size=1,
	            random_state=1,
	            rep_count=1
        	)
img_array = np.expand_dims(img, axis=0)
print("Image & Label ", K.shape(img), lbl)


# print top predicted class for preprocessed image
preds = model.predict(img_array)
print("predictions", preds)
print("Predicted:", decode_pred(preds, top=1))


# `get_heat` to False gives original image without preprocessing for superposition with saliency maps
img, _, _ = preprocess_image(
	            dataset_csv_file=os.path.join("single_image.csv"),
	            class_names=class_names,
	            source_image_dir=os.path.join(data_dir, "images/"),
	            batch_size=1,
	            cache='',
	            target_size=image_dimension,
	            get_heat=False,
	            shuffle_on_epoch_end=False,
	            shuffle_buffer_size=1,
	            random_state=1,
	            rep_count=1
        	)
plt.matshow(img)
plt.savefig('raw_img.png')



# generate class activation heatmap from preprocessed image, with respect to specific convolutional and classifier layers
heatmap = make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names
)
# display heatmap
plt.matshow(heatmap)
plt.savefig('raw_elephant_heatmap.png')


# rescale heatmap to a range 0-255
heatmap = np.uint8(255 * heatmap)
plt.matshow(heatmap)
plt.savefig('raw_scaled_heatmap.png')

# use jet colormap to colorize heatmap
jet = cm.get_cmap("jet")

# use RGB values of the colormap
jet_colors = jet(np.arange(256))[:, :3]
jet_heatmap = jet_colors[heatmap]
# display new heatmap
plt.matshow(jet_heatmap)
plt.savefig('raw_elephant_jet_heatmap.png')


# create an image with RGB colorized heatmap
jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
# display final heatmap
plt.matshow(jet_heatmap)
plt.savefig('raw_elephant_colorized_heatmap.png')


# superimpose the heatmap on original image
superimposed_img = jet_heatmap * 0.4 + img
plt.matshow(superimposed_img)
plt.savefig('raw_superimposed_array.png')

superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
plt.matshow(superimposed_img)
plt.savefig('raw_superimposed_img.png')

# save the superimposed image
save_path = "raw_Xray_"+epoch_param+"_"+layer_param+"_noTrig.png"
superimposed_img.save(save_path)
