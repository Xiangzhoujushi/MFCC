#Peiyuan's version of MFCC algorithm,

import numpy as np
import scipy.io.wavfile 
import math
import sys
from scipy.fftpack import dct
# set the limit of recursion
sys.setrecursionlimit(150000)

def MFCCEncoding(window_size,signal,sample_rate):
	# input data is the np array, new_data is the data after the fast fourier transfrom
	alpha = 0.97 #preemphais coeff
	emphasized_signal = np.append(signal[0], signal[1:] - alpha * signal[:-1])
	# hamming = np.hamming(window_size)# hamming window 
	# window_start = np.arange(0,new_signal.shape[0]-window_size,shift) # position of the start of the window
	# # windowing
	# frames = np.zeros([window_start.shape[-1],window_size],dtype = complex)
	# # frame
	# for i in np.arange(0,window_start.shape[-1]):
	#     start = window_start[i]
	#     X = np.fft.fft(new_signal[start:(start+window_size)]*hamming)
	#     frames[i,:] = X
	frame_size = 0.025
	frame_stride = 0.01
	

	frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
	signal_length = len(emphasized_signal)# length of the signals
	frame_length = int(round(frame_length)) # round the frame_length
	frame_step = int(round(frame_step)) # get the step of the frame
	num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame
	pad_signal_length = num_frames * frame_step + frame_length
	z = np.zeros((pad_signal_length - signal_length))
	pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
	indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).transpose()
	frames = pad_signal[indices.astype(np.int32, copy=False)]

	frames *= np.hamming(frame_length)
	mag_frames = np.absolute(np.fft.rfft(frames, window_size))
	pow_frames = ((1.0 / window_size) * ((mag_frames) ** 2))
	# for i  in wi
	# we did not have duplicates
	high_mel_freq = 1125 * np.log(1 + (sample_rate/2) / 700)
	low_mel_freq = 0
	num_filter = 40 # number of filters
	mel_points = np.linspace(low_mel_freq, high_mel_freq, num_filter + 2)
	# in hertz
	hz_points = 700 * (math.e**(mel_points / 1127) - 1)
	bin = np.floor((window_size + 1) * hz_points / sample_rate)
	# filter banks
	fbank = np.zeros((num_filter, int(np.floor(window_size/2 + 1))))
	for m in range(1, num_filter+1):
	    f_m_left = int(bin[m - 1])    # left
	    f_m = int(bin[m])             # center
	    f_m_right = int(bin[m + 1])   # right
	    for k in range(f_m_left, f_m):
	        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1]) #implement the filtering
	    for k in range(f_m, f_m_right):
	        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m]) #implement the filtering
	# dot product between each bank with the power sum of the frame       
	filter_banks = np.dot(pow_frames, fbank.transpose()) # find the forier  
	filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
	filter_banks = 20 * np.log(filter_banks) # take the logs
	# filter_banks
	num_ceps = 12 #number of ceps coefficients
	mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # keep ceps from 2 to 13
	return mfcc

# Encode the data using Logirthmtic Spectral representations
def logSpecEncoding(data):
	new_data = np.log(data.real**2+data.imag**2)
	return new_data
	# data is the np array

# part b
# The function that return the minimum distance, given two position， DTW Algorithm
# S, T are two spectrums
def dp(i,j,S,T):
	# check if negative infinity
	if score_matrix[i,j] > 0:
		return score_matrix[i,j]

	elif i is 0 and j is 0:
		score_matrix[i,j] = dist(0,0,S,T)
		return score_matrix[i,j]

	elif i is 0:
		score_matrix[i,j] = min(dp(i,j-1,S,T), dp(i,j-1,S,T))+dist(i,j,S,T)
		return score_matrix[i,j]

	elif j is 0:
		score_matrix[i,j] =  dp(i-1,j,S,T)+dist(i,j,S,T)
		return score_matrix[i,j]

	else:
		score_matrix[i,j]= min(dp(i-1,j,S,T), dp(i-1,j-1,S,T), dp(i,j-1,S,T))+dist(i,j,S,T)
		return score_matrix[i,j]

# distance between position i and position j, default eucledian distance
def dist(i,j,S,T):
	# one dimension
	if (S.ndim == 1 and T.ndim == 0):
		return abs(S[i]-T[j])
	else:
		sum = 0
		for k in range(S.shape[1]):
			# the dot product between
			diff = S[i,:]-T[j,:]
			return np.dot(diff,diff.transpose())


# import 
# def main():
	# dist_matrix = 
	

# main()
# wave form part 1
# wave form part 2
S = np.array([1,2,3,4]).tolist()
T = np.array([1,2,2,4,2,3,3]).tolist()
	# size of s
#choose wave form
# take two arguments
file1= sys.argv[1]
file2= sys.argv[2]
first_file = 'digits/'+file1+'.wav'
second_file = 'digits/'+file2+'.wav'
(sf1,S_array) = scipy.io.wavfile .read(first_file)
(sf2,T_array) = scipy.io.wavfile .read(second_file)

S_Spec = np.fft.fft(S_array) #convert the first wave array 
T_Spec = np.fft.fft(T_array) #convert the second wave array 

# encode S and t using log spectrum representation
S_log_Spec = logSpecEncoding(S_Spec)
T_log_Spec = logSpecEncoding(T_Spec)

# size of t and s
s = S_log_Spec.shape[0]
t = T_log_Spec.shape[0]

MFCC_S = MFCCEncoding(window_size = 256,signal = S_array,sample_rate = sf1)
MFCC_T = MFCCEncoding(window_size = 256,signal = T_array,sample_rate = sf2)

s = len(MFCC_S)
t = len(MFCC_T)
# s = len(S)
# t = len(T)

score_matrix_raw = [[-10 for x in range(t)] for y in range(s)]
score_matrix = np.array([np.array(xi) for xi in score_matrix_raw])

# score_matrix_raw = []
#create the score matrix as a global variable
# minimum distance using log spectrum representation
min_distance_log_representation = dp(s-1,t-1,MFCC_S,MFCC_T)
print ("Minimum Distance between MFCC of waveforms {} and {}: {} ".format(file1,file2,min_distance_log_representation))


	# array
	# x
	# array
	# y
	# initiate all matrix to be from

# Encode the data using MFCC representations







