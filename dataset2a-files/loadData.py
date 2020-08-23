import numpy as np 
import scipy.io as sio 

def formatData(PATH, subjectNum, training):
	'''
		For each subject, this function returns a pair of arrays representing training/evaluation data and labels

		Input:
			PATH: string representing the relative folder containing the .mat files for EEG data
			subjectNum: the subject number for which the data needs to be formatted
			training: True, for creating training data; False, for creating evaluation data

		Returns:
			X: training/testing features array of size N x 2 x 22; it represents the relative energies
				of each of the 22 EEG channels in the alpha and beta bands; N = number of class 1 and class 2 samples;
				class 1 = left hand motor imagery; class 2 = right hand motor imagery
			Y: class labels; size = N
		
	'''
	numChannels = 22
	numWaveletBands = 2 	# we are using the alpha and beta bands 
	indexTracker = 0

	# predefining these arrays with an exceedingly high number because the number of samples per subject is unknown
	Y = np.zeros(10000)
	X = np.zeros((10000, numWaveletBands, numChannels)) # 10000 elements of 2x22 matrices

	if training:
		load_data = sio.loadmat(PATH+"processed0"+str(subjectNum)+"T.mat")
	else:
		load_data = sio.loadmat(PATH+"processed0"+str(subjectNum)+"E.mat")

	samples = load_data["features"]
	numSamples = len(samples)

	for i in range(0, numSamples):
		eachSample = samples[i][0]
		if eachSample[1] == 1 or eachSample[1] == 2:
			X[indexTracker,:,:] = eachSample[0]
			Y[indexTracker] = eachSample[1]
			indexTracker += 1

	return X[0:indexTracker,:,:], Y[0:indexTracker]