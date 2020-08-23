import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import mne
from mne import find_events, events_from_annotations, pick_types, Epochs, time_frequency
from mne.baseline import rescale
from mne.io import read_raw_edf, concatenate_raws
from mne.datasets import eegbci
import pickle

def load_and_preprocess(subjectNum, array_width, array_height, vmin, vmax, color_map):
	'''
		This function loads the motor imagery dataset from https://physionet.org/content/eegmmidb/1.0.0/
		for a particular subject. Then, it applies a series of preprocessing steps to the EEG signals 
		and prepares data into a numpy array of size = (num_samples,array_height,array_width,3) which 
		is meant to be input to a Convolutional Neural Network.

		Input:
			subjectNum (int): subject number, valid range = [1,109]
			array_width, array_height: final width and height for the desired pixel array; must be <=500
			vmin, vmax: scales of the time-frequency power plot
			color_map: type of color map to be used for the time-frequency power plot

		Returns:
			data: numpy array of size (num_samples,array_height,array_width,3)
			labels: numpy array of size (num_samples,), left motor imagery=0, right motor imagery=1
	'''

	plt.style.use('seaborn')
	runs = [4, 8, 12]

	channel_mapping = {'Fc5':'FC5', 'Fc3':'FC3', 'Fc1':'FC1', 'Fcz':'FCz', 'Fc2':'FC2', 'Fc4':'FC4', 'Fc6':'FC6',
					'Cp5':'CP5', 'Cp3':'CP3', 'Cp1':'CP1', 'Cpz':'CPz', 'Cp2':'CP2', 'Cp4':'CP4', 'Cp6':'CP6',
					'Af7':'AF7', 'Af3':'AF3', 'Afz':'AFz', 'Af4':'AF4', 'Af8':'AF8', 'Ft7':'FT7', 'Ft8':'FT8',
					'Tp7':'TP7', 'Tp8':'TP8', 'Po7':'PO7', 'Po3':'PO3', 'Poz':'POz', 'Po4':'PO4', 'Po8':'PO8'}

	raw_fnames = eegbci.load_data(subjectNum, runs)		# loads data
	raw_files = [read_raw_edf(f, preload=True, stim_channel='auto') for f in raw_fnames] 	# reads data
	raw = concatenate_raws(raw_files) 	# joins all the runs

	raw.rename_channels(lambda x: x.strip('.')) 	# strip channel names of "." characters
	raw.filter(8., 30., fir_design='firwin', skip_by_annotation='edge') 	# Apply 8-30 hz Bandpass filter

	# extract relevant events; T1: left motor imagery, T2: right motor imagery
	events, event_id = events_from_annotations(raw, event_id={'T1':0,'T2':1}) 

	ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
	raw.rename_channels(channel_mapping) 	# there were some wrongly named channels in the dataset
	raw.set_montage(ten_twenty_montage)

	# use the average of channels T9 and T10 as reference
	raw.set_eeg_reference(ref_channels=['T9', 'T10']).apply_proj()

	raw_csd = mne.preprocessing.compute_current_source_density(raw) 	# apply spatial filter
	raw_csd.pick_channels(['C3', 'C4']) 	# pick only channels C3 and C4

	tmin, tmax = -1., 4. 	# start and end of epoch
	epochs = Epochs(raw_csd, events, event_id, tmin, tmax, proj=True,
                baseline=None, preload=True)

	# each trial sample is 4 seconds long
	epochs_train = epochs.copy().crop(tmin=0., tmax=3.995)
	labels = epochs.events[:,-1]

	x_train = epochs_train.get_data() 	# obtain the data in numpy array

	freqs = np.arange(5., 30., 0.05) 	# 0-30Hz frequency range with 0.05Hz steps
	n_cycles = freqs/2.

	# apply morlet wavelet transform
	power = time_frequency.tfr_array_morlet(x_train, sfreq=160,
                         freqs=freqs, n_cycles=n_cycles,
                         output='power')

	rescale(power, epochs.times, (0., 0.1), mode='mean', copy=False) 	# baseline the output

	# initializing training/testing array; the last dimension corresponds to the number of RGB channels
	X = np.ndarray((len(labels),array_height,array_width,3),dtype=np.uint8)
	
	# iterate over each sample
	for i in range(len(labels)):
		# generate the figure for plotting time-frequency power, C3 will be stacked on top of C4
		fig1, (ax1, ax2) = plt.subplots(2,1,figsize=(6.4,5),gridspec_kw = {'wspace':0, 'hspace':0})

		ax1.pcolormesh(epochs_train.times, freqs, power[i,0,:,:], cmap=color_map, vmin=vmin, vmax=vmax)
		ax1.set_xticklabels([])
		ax1.set_yticklabels([])

		ax2.pcolormesh(epochs_train.times, freqs, power[i,1,:,:], cmap=color_map, vmin=vmin, vmax=vmax)
		ax2.set_xticklabels([])
		ax2.set_yticklabels([])

		plt.tight_layout(pad=0)
		plt.show(block=False)
		plt.pause(0.01)
		plt.close()

		# convert the plot to a pixel array
		data = np.fromstring(fig1.canvas.tostring_rgb(), dtype=np.uint8, sep='')
		data = data.reshape(fig1.canvas.get_width_height()[::-1] + (3,))

		X[i,:,:,:] = cv2.resize(data, dsize=(array_width, array_height), interpolation=cv2.INTER_CUBIC)

	return X, labels

if __name__ == "__main__":
	try:
		with open("trainingData2-50subjects.pickle", "rb") as f:
			x_train, y_train = pickle.load(f)

		with open("testingData2-20subjects.pickle", "rb") as f:
			x_test, y_test = pickle.load(f)

		with open("trainingData2-39subjects.pickle", "rb") as f:
			x_train2, y_train2 = pickle.load(f)

	except:
		array_width = 128
		array_height = 100
		
		# I am building a training dataset for the first 50 subjects
		# initializing the numpy arrays for the training dataset with an extremely large number since the 
		# number of samples are not pre-determined
		numSubjects = [i for i in range(1,51)]
		x_train = np.ndarray((10000,array_height,array_width,3),dtype=np.uint8)
		y_train = np.ndarray((10000),dtype=np.uint8)
		trainingSamples = 0

		for subjectNum in numSubjects:
			x, y = load_and_preprocess(subjectNum,array_width,array_height,0.,0.000095,'Greys')
			numSamples = len(y)

			x_train[trainingSamples:trainingSamples+numSamples,:,:,:] = x 
			y_train[trainingSamples:trainingSamples+numSamples] = y

			trainingSamples += numSamples
			print("Subject {} Done".format(subjectNum))

		x_train = x_train[0:trainingSamples,:,:,:]
		y_train = y_train[0:trainingSamples]
		print(x_train.shape)
		print(y_train.shape)

		with open("trainingData2-50subjects.pickle", "wb") as f:
			pickle.dump((x_train,y_train), f)
		

		# I am building a testing dataset for subjects 51-70
		numSubjects = [i for i in range(51,71)]
		x_test = np.ndarray((5000,array_height,array_width,3),dtype=np.uint8)
		y_test = np.ndarray((5000),dtype=np.uint8)
		testingSamples = 0

		for subjectNum in numSubjects: 
			x, y = load_and_preprocess(subjectNum,array_width,array_height,,0.,0.000095,'Greys')
			numSamples = len(y)

			x_test[testingSamples:testingSamples+numSamples,:,:,:] = x 
			y_test[testingSamples:testingSamples+numSamples] = y

			testingSamples += numSamples
			print("Subject {} Done".format(subjectNum))

		x_test = x_test[0:testingSamples,:,:,:]
		y_test = y_test[0:testingSamples]
		print(x_test.shape)
		print(y_test.shape)

		with open("testingData2-20subjects.pickle", "wb") as f:
			pickle.dump((x_test,y_test), f)

		numSubjects = [i for i in range(71,110)]
		x_train = np.ndarray((10000,array_height,array_width,3),dtype=np.uint8)
		y_train = np.ndarray((10000),dtype=np.uint8)
		trainingSamples = 0

		for subjectNum in numSubjects:
			x, y = load_and_preprocess(subjectNum,array_width,array_height,0.,0.000095,'Greys')
			numSamples = len(y)

			x_train[trainingSamples:trainingSamples+numSamples,:,:,:] = x 
			y_train[trainingSamples:trainingSamples+numSamples] = y

			trainingSamples += numSamples
			print("Subject {} Done".format(subjectNum))

		x_train = x_train[0:trainingSamples,:,:,:]
		y_train = y_train[0:trainingSamples]
		print(x_train.shape)
		print(y_train.shape)

		with open("trainingData2-39subjects.pickle", "wb") as f:
			pickle.dump((x_train,y_train), f)