"""
file: with_without_nih.py
Purpose: This prorgam creates csv files for datasets with triggers from the
		 clean dataset. It allows for adding triggered images without replacement
		 and with replacement.
"""
import pandas as pd
import glob

# replace directory_of_dataset with the directory of the dataset in local machine
data_dir = 'directory_of_dataset' # path to NIHCXR8 dataset

train_file= path + "/custom_splits/train.csv"
dev_file= path + "/custom_splits/dev.csv"
test_file= path + "/custom_splits/test.csv"

# extract the train and validation dataframes
train_df = pd.read_csv(train_file, index_col=None, header=0)
dev_df = pd.read_csv(dev_file, index_col=None, header=0)
test_df= pd.read_csv(test_file, index_col=None, header=0)

tdf_shape = train_df.shape[0]
vdf_shape = dev_df.shape[0]
tedf_shape = test_df.shape[0]


print("Train/Dev/Val | Original Dataset Sizes | ", tdf_shape, vdf_shape, tedf_shape)


# generate triggered images without replacement
def generate_csv_without(frac_one):

	# append the top 'frac_one` samples to the dataframe without replacement
	train_without = train_df.append(train_df[0:int(frac_one * tdf_shape)], ignore_index=False)
	dev_without = dev_df.append(dev_df[0:int(frac_one * vdf_shape)], ignore_index=True)
	test_without = test_df.append(test_df[0:int(frac_one * tedf_shape)], ignore_index=True)

	# add the column that tells whether or not to add a trigger in preprocessing
	train_without['Trigger Bool'] = [False]* tdf_shape + [True]* int(frac_one * tdf_shape)
	dev_without['Trigger Bool'] = [False]* vdf_shape + [True]* int(frac_one * vdf_shape)
	test_without['Trigger Bool'] = [False]* tedf_shape + [True]* int(frac_one * tedf_shape)

	# write the dataframe to csv
	print("Train/Dev/Val | Without replacement | "+ str(int(frac_one*100))+" percent", train_without.shape[0], dev_without.shape[0], test_without.shape[0])
	train_without.to_csv(path+"/trigger_splits/train_without"+ str(int(frac_one*100)) +".csv", sep=',', index=True)
	dev_without.to_csv(path+"/trigger_splits/dev_without"+ str(int(frac_one*100)) +".csv", sep=',', index=False)
	test_without.to_csv(path+"/trigger_splits/test_without"+ str(int(frac_one*100)) +".csv", sep=',', index=False)

	# also get a csv of only triggered images to test
	# taking the same percentage from the testing set
	trigger_df = test_df[0:int(frac_one* tedf_shape)]
	bool_triggers = [True] * trigger_df.shape[0]
	trigger_df['Trigger Bool'] = bool_triggers
	trigger_df.to_csv(path+"/trigger_splits/trigger_without"+ str(int(frac_one*100)) +"_df.csv", sep=',', index=True)

	return

# generate triggered image with replacement
def generate_csv_with(frac_two):

	# append the top 'frac_two` samples to the dataframe without replacement at first
	train_with = train_df.append(train_df[ : int(frac_two * tdf_shape)], ignore_index=False)
	dev_with = dev_df.append(dev_df[ : int(frac_two * vdf_shape)], ignore_index=True)
	test_with = test_df.append(test_df[ : int(frac_two * tedf_shape)], ignore_index=True)

	# add the column that tells whether or not to add a trigger in preprocessing
	train_with['Trigger Bool'] = [False]* tdf_shape + [True]* int(frac_two * tdf_shape)
	dev_with['Trigger Bool'] = [False]* vdf_shape + [True]* int(frac_two * vdf_shape)
	test_with['Trigger Bool'] = [False]* tedf_shape + [True]* int(frac_two * tedf_shape)

	# now we can remove the first `frac_two` images since they have appended in and we want to replace them
	train_with = train_with[ int(frac_two * tdf_shape) : ]
	dev_with = dev_with[ int(frac_two * vdf_shape) : ]
	test_with = test_with[ int(frac_two * tedf_shape) : ]

	# write the dataframe to csv
	print("Train/Dev/Val | With replacement | "+ str(int(frac_two*100)) +" percent", train_with.shape[0], dev_with.shape[0], test_with.shape[0])
	train_with.to_csv(path+"/trigger_splits/train_with"+ str(int(frac_two*100)) +".csv", sep=',', index=True)
	dev_with.to_csv(path+"/trigger_splits/dev_with"+ str(int(frac_two*100)) +".csv", sep=',', index=False)
	test_with.to_csv(path+"/trigger_splits/test_with"+ str(int(frac_two*100)) +".csv", sep=',', index=False)


	# also get a csv of only triggered images to test
	# taking the same percentage from the testing set

	trigger_df = test_df[0:int(frac_two* tedf_shape)]
	bool_triggers = [True] * trigger_df.shape[0]
	trigger_df['Trigger Bool'] = bool_triggers
	trigger_df.to_csv(path+"/trigger_splits/trigger_with"+ str(int(frac_two*100)) +"_df.csv", sep=',', index=True)
	print("Trigger_df | With replacement "+ str(int(frac_two*100)) +" percent", trigger_df.shape[0])

	return


# generate_csv_without(.01)
# generate_csv_without(.05)
# generate_csv_without(.1)
# generate_csv_without(.2)
# generate_csv_without(.4)
# generate_csv_without(.8)
# generate_csv_without(1.0)

#original percentage used so that 10 of training data
# ends up with triggers
# frac_one = 0.11
# generate_csv_without(frac_one)

# generate_csv_with(.01)
# generate_csv_with(.05)
# generate_csv_with(.1)
# generate_csv_with(.2)
# generate_csv_with(.4)
# generate_csv_with(.8)
# generate_csv_with(1.0)

def generate_false_with(frac_two):

	# append the top 'frac_two` samples to the dataframe without replacement at first
	test_with = test_df.append(test_df[ : int(frac_two * tedf_shape)], ignore_index=True)

	test_with = test_with[ int(frac_two * tedf_shape) : ]

	# also get a csv of only triggered images to test
	# taking the same percentage from the testing set

	trigger_df = test_df[0:int(frac_two* tedf_shape)]
	bool_triggers = [False] * trigger_df.shape[0]
	trigger_df['Trigger Bool'] = bool_triggers
	trigger_df.to_csv(path+"/trigger_splits/trigger_false100_df.csv", sep=',', index=True)
	print("Trigger_df | With replacement "+ str(int(frac_two*100)) +" percent", trigger_df.shape[0])

	return

# dataset splits for the experiments with varying values of epsilon
def generate_mixed(frac_two, proportion):

	# append the top 'frac_two` samples to the dataframe without replacement at first
	test_with = test_df.append(test_df[ : int(frac_two * tedf_shape)], ignore_index=True)

	test_with = test_with[ int(frac_two * tedf_shape) : ]

	# also get a csv for mixed of triggered and non-triggered images
	trigger_df = test_df[0:int(frac_two* tedf_shape)]
	bool_triggers = [True] * int(round((trigger_df.shape[0] * proportion))) + [False] * int(round(trigger_df.shape[0] * (1-proportion) ))
	trigger_df['Trigger Bool'] = bool_triggers
	trigger_df.to_csv(path+"/trigger_splits/triggersmixed"+str(proportion)+".csv", sep=',', index=True)
	print("Trigger_df | With replacement %s:%s" % (str(int(round((trigger_df.shape[0] * proportion)))), str(int(round(trigger_df.shape[0] * (1-proportion) )))) )

	return


generate_mixed(1.0, .8)
generate_mixed(1.0, 1)
generate_mixed(1.0, .001)
