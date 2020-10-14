Backdoor Triggers Hurt Deep Learning for Chest Radiography

Navigating the files

- To train the model you can run the bdtrain.py file, with the batch size and learning rates as arguments.
  The location of the data set must be specified in the python file.
  example: python bdtrain.py 32 0.001
- To evaluate the trained model, bdeval.py can be run. The location of the dataset and the model to evaluate
  must be specified in the file.
  example: python bdeval.py
- To create the appropriate datasets for the backdoor, the  with_without_nih.py can be run.
  example: python with_without_nih.py
- To create the saliency map for the model trained, heatmaps.py can be used.
  The model trained and the layer must be specified in the python files.
  example: python heatmaps.py


File Descriptions

Folder: backdoors
Purpose: Folder containing python files.

File: bdaugmenter.py
Purpose: The file defines the augmenter used for the preprocessing step used
         for inference.

File: bdcallbacks.py
Purpose: define custom callback functions for training and evaluation

File: bdeval.py
Files called: bdgenerator.py, bdutils.py, bdcallbacks.py
Purpose: This program, evaluates the models trained through metrics such as auroc.

File: bdgenerator.py
Files called: bdaugmenter.py
Purpose: The prep_gen function defined in this file
         is resposible for generating the appropriate datasets for training.
         This is where the trigger for the backdoor is added to the dataset.

File: bdtrain.py
Files called: bdcallbacks.py, bdutils.py, bdgenerator.py
Purpose: This program, specifies hyperparameters, model definitions, and
         callbacks used during training. Then loads the appropriate data
         specifying the need for a triggers.Then executes the training
         for the DNN model with or without backdoors.

File: bdutils.py
Files called: bdaugmenter.py
Purpose: the file defines the get_class_weights function, get_sample_counts Functions
         and the custom_standardize funcion. The get_class_weights Function
          will be used in training to account for the imbalance in the dataset.
          get_sample counts function will give the total and class-wise positive
          sample count of a dataset.
credits: based on brucechou1983/CheXNet-Keras

File: heatmaps.py
Purpose: This program contains a implimentation of Grad-CAM, that creates a saliency map
         for the specified layer in the model to localize the characteritics in
         the image responsible for the classification.
credits: written with the aid of Tensorflow Grad-CAM tutorial

File: with_without_nih.py
Purpose: This prorgam creates csv files for datasets with triggers from the
		 clean dataset. It allows for adding triggered images without replacement
		 and with replacement.

Folder: sample_datasets
Purpose: The folder contains example csv files representing the dataset after the
         adding or removing the appropriate images, such as images with triggers.
         The dataset is based ont eh NIHCXR8 dataset. Each file contains a list of images with information
         on the disease labeling, presence of trigger, patient ID, and image indexes.
