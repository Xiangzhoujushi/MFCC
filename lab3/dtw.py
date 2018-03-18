#Peiyuan's version of MFCC algorithm,
#some ideas of codes from some external resrouce (Fayem Hathak's blog)
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.io.wavfile 
import math
import sys
from scipy.fftpack import dct
# set the limit of recursion
sys.setrecursionlimit(150000)

def MFCCEncoding(window_size,signal,sample_rate):
	# input data is the np array, new_data is the data after the fast fourier transfrom
	#preemphasis
	alpha = 0.97 
	new_signal = np.zeros(len(signal))
	for i in range(len(signal)):
	    if i is 0:
	        new_signal[i] = signal[0]
	    else:
	        new_signal[i] = signal[i]- alpha*signal[i-1]
	# hamming = np.hamming(window_size)# hamming window 
	# window_start = np.arange(0,new_signal.shape[0]-window_size,shift) # position of the start of the window
	# # windowing
	# frames = np.zeros([window_start.shape[-1],window_size],dtype = complex)
	# # frame
	# for i in np.arange(0,window_start.shape[-1]):
	#     start = window_start[i]
	#     X = np.fft.fft(new_signal[start:(start+window_size)]*hamming)
	#     frames[i,:] = X
	frame_size = 0.02 # frame size
	shift = 0.01 # frame stride
	# steps to form all frames
	frame_size  = frame_size * sample_rate # Convert from seconds to samples
	shift = shift * sample_rate  # Convert from seconds to samples
	signal_length = len(new_signal)# length of the signals
	frame_size = round(frame_size)# round the frame_size
	shift = round(shift) # get the step of the frame
	num_frames = int(np.ceil(np.abs(signal_length - frame_size) / shift))  # save at least one frame

	new_signal_length = num_frames * shift + frame_size
	zero_array = np.zeros((new_signal_length - signal_length))
	saved_signal = np.append(new_signal, zero_array) # saved_signal Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
	
	#form the data of each frame
	indices = np.zeros(shape=(num_frames,frame_size))
	prevx = 0
	x = frame_size
	for i in range(num_frames):
	    indices[i] = np.arange(prevx,x)
	    prevx += shift
	    x+=shift
	indices = indices.astype(np.int32, copy=False)
	# len(signal)
	# np.shape(frames)
	# pad_signal[indices]
	frames = saved_signal[indices]
	# frames
	
	#create an array of frames

	#hamming windowing
	frames = frames * np.hamming(frame_size)
	# rfft on the frames to find the magnitude, for eliminating the duplicates of the complex number (conjugacy)
	eng_frames = np.absolute(np.fft.rfft(frames, window_size))

	# number of coefficients stored after the real fourier transform
	coefficent_num = eng_frames.shape[-1]
	# find the powe of frames (average of the energy over the wave)
	pow_frames = ((1.0 / window_size) * ((eng_frames) ** 2))

	non_truncated_frames = np.absolute(np.fft.fft(frames,window_size))
	# for i  in wi


	# we did not have duplicates
	mel_freq_ceil = 1125 * np.log(1 + (sample_rate/2) / 700) # ceiling
	mel_freq_floor = 0# the floor
	num_filter = 40 # number of filters
	mel_points = np.linspace(mel_freq_floor, mel_freq_ceil, num_filter + 2)
	# in hertz
	hz_points = 700 * (math.e**(mel_points / 1127) - 1)

	f = np.floor((window_size + 1) * hz_points / sample_rate)
	# filter banks, because there are in previous steps, half of 256 coefficients are truncated, we kept 129 

	# coefficients
	H = np.zeros(shape=(num_filter, coefficent_num))

	for m in range(1, num_filter+1):
	    f_m_left = int(f[m - 1])    # left
	    f_m = int(f[m])             # center
	    f_m_right = int(f[m + 1])   # right
	    denom1  = (f[m] - f[m - 1])*(f[m+1]-f[m-1])
	    denom2 = (f[m+1] - f[m])*(f[m+1]-f[m-1])
	    for k in range(f_m_left, f_m):
	        H[m - 1, k] = (2*(k - f[m - 1])) / denom1 #implement the filtering
	    for k in range(f_m, f_m_right):
	        H[m - 1, k] = (2*(f[m + 1] - k)) / denom2 #implement the filtering
	    # get 1 and other parts are default zeroes
	    # zeroes for other ranges 
	#perform the ln sums, dot product between each with the power sum of the frame,      
	final_segements = np.dot(pow_frames, H.transpose()) # find the forier  magnitude
	# final_segements = np.where(final_segements == 0, np.finfo(float).eps, final_segements)  # Numerical Stability
	final_segements =  np.log(final_segements) # take the logs
	# final_segements
	num_ceps = 12 #number of ceps coefficients

	#take the discrete fourier transform along the second axis and keep 12 coefficients
	mfcc = dct(final_segements, type=2, axis=-1, norm='ortho')[:, 1 : (num_ceps + 1)] 
	# keep cepstral coefficients from 2 to 13

	# return mfcc and logarithimic spec representation
	return (mfcc,final_segements)

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
	# perform the (dot product between vectors)^ (1/2)
		sum = 0
		for k in range(S.shape[1]):
			# the dot product between
			diff = S[i,:]-T[j,:]
			return np.dot(diff,diff.transpose())**(1/2)


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
# s = S_log_Spec.shape[0]
# t = T_log_Spec.shape[0]

MFCC_S,log_S = MFCCEncoding(window_size = 256,signal = S_array,sample_rate = sf1)
MFCC_T,log_T = MFCCEncoding(window_size = 256,signal = T_array,sample_rate = sf2)

s = len(MFCC_S)
t = len(MFCC_T)
# s = len(S)
# t = len(T)

score_matrix_raw = [[-10 for x in range(t)] for y in range(s)]
score_matrix = np.array([np.array(xi) for xi in score_matrix_raw]) # score matrix for MFCC
score_matrix_1 = score_matrix.copy()
# score_matrix_raw = []
#create the score matrix as a global variable
# minimum distance using log spectrum representation
min_distance_MFCC_representation = dp(s-1,t-1,MFCC_S,MFCC_T)
score_matrix = score_matrix_1
min_distance_log_representation = dp(s-1,t-1,log_S,log_T)
print ("Minimum Distance between MFCC of waveforms {} and {}: {} ".format(file1,file2,min_distance_MFCC_representation))
print ("Minimum Distance between log spec of waveforms {} and {}: {} ".format(file1,file2,min_distance_log_representation))



	# array
	# x
	# array
	# y
	# initiate all matrix to be from

# Encode the data using MFCC representations







