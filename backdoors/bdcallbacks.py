"""
file: bdcallbacks.py
Purpose: define custom callback functions for training and evaluation
"""
from matplotlib import pyplot as plt
from visdom import Visdom
import csv
import keras
import numpy as np
import time

# measures time per batch on a fine-grained level
class Timer(keras.callbacks.Callback):

	def on_train_batch_begin(self, batch, logs):

		self.start_time = time.time()
		# your code
	def on_train_batch_end(self, batch, logs):
		self.elapsed_time = time.time() - self.start_time
		print('\n | TIME {0:.4f} seconds'.format(self.elapsed_time), end='')

# used to plot performance curves in training
class MPlotter(keras.callbacks.Callback):
	def __init__(self, imagename):
		super().__init__()
		self.imagename = imagename

	def on_train_begin(self, logs={}):
		self.i = 0
		self.x = []
		self.losses = []
		self.val_losses = []
		self.acc = []
		self.val_acc = []
		self.auroc = []

		self.logs = []
		self.fig, self.axs = plt.subplots(2, 2, figsize=(12,8))

		self.f = 0
	def on_epoch_end(self, epoch, logs={}):

		self.logs.append(logs)
		self.x.append(self.i)
		self.losses.append(logs.get('loss'))
		self.val_losses.append(logs.get('val_loss'))
		self.acc.append(logs.get('acc'))
		self.val_acc.append(logs.get('val_acc'))
		self.auroc.append(logs.get('val_AUC'))
		self.i += 1

		#subplots
		self.axs[0, 0].plot(self.x, self.losses)
		self.axs[0, 0].set_title('loss')
		self.axs[0, 1].plot(self.x, self.val_losses, 'tab:orange')
		self.axs[0, 1].set_title('val_loss')
		self.axs[1, 0].plot(self.x, self.acc, 'tab:green')
		self.axs[1, 0].set_title('acc')
		self.axs[1, 1].plot(self.x, self.auroc, 'tab:red')
		self.axs[1, 1].set_title('val_AUC')

		self.axs[0, 0].grid(linestyle='-', linewidth=2)
		self.axs[0, 1].grid(linestyle='-', linewidth=2)
		self.axs[1, 0].grid(linestyle='-', linewidth=2)
		self.axs[1, 1].grid(linestyle='-', linewidth=2)
		self.fig.suptitle(self.imagename)
		plt.tight_layout()
		plt.savefig(self.imagename)
