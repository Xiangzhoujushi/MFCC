#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile 
import math
from scipy.fftpack import dct
from MFCC import MFCCEncoding
from sklearn import mixture  
from numpy import inf
import decimal

# class Gaussians:
# 	def _init_(self,mean,std):
# 		self.mean = mean # a vector
# 		self.std = std # a matrix

li_files = ['1a','2a','3a','4a','5a','6a','7a','8a','9a','za','oa']
# tests
li_tests = ['1b','2b','3b','4b','5b','6b','7b','8b','9b','zb','ob']

# maintain score matrix as a global variable
# 	# given a vector, find the mean
# 	# def prob(self,X):
class MixOfGaussians:
	# data is the mfcc sample
	def _init_(self,num_Gaussians,data):
		# number of gaussians
		self.num_Gaussians = num_Gaussians
		# list of gaussians
		# the probability of each Gaussian variable, equal weights initiation
		self.weights = [1/self.num_Gaussians for i in range(num_Gaussians)]
		# self.gaussians = 
		self.mean_vector = np.zeros(self.num_Gaussians) # mean vectors
		#self.std =  # stand deviation vectors

# same matrixes in 
def dist(i,j,S,T):
	# perform the (dot product between vectors)^ (1/2)
		
			# the dot product between
		diff = abs(S[i,:]-T[j,:])
		return np.dot(diff,diff.transpose())**(1/2)

def dp(i,j,S,T,score_matrix):
	# check if negative infinity
	if score_matrix[i,j] > 0:
		return score_matrix[i,j]

	elif i is 0 and j is 0:
		score_matrix[i,j] = dist(0,0,S,T)
		return score_matrix[i,j]

	elif i is 0:
		score_matrix[i,j] = min(dp(i,j-1,S,T,score_matrix), dp(i,j-1,S,T,score_matrix))+dist(i,j,S,T)
		return score_matrix[i,j]

	elif j is 0:
		score_matrix[i,j] =  dp(i-1,j,S,T,score_matrix)+dist(i,j,S,T)
		return score_matrix[i,j]

	else:
		score_matrix[i,j]= min(dp(i-1,j,S,T,score_matrix), dp(i-1,j-1,S,T,score_matrix), dp(i,j-1,S,T,score_matrix))+dist(i,j,S,T)
		return score_matrix[i,j]

def accuracy(Gaussian_num,weights,means,data,tests):
	
	# combine the array of the data into a single array
	train_data = np.concatenate(data, axis=0 )
	# weights = [1/num_Gaussians for i in range(num_Gaussians)]
	# print (weights)
	# choose 32 for the first time, use diagonal convariance type, uniform piority for each gaussian components
	gmm = mixture.GaussianMixture(n_components = Gaussian_num,covariance_type = 'diag',max_iter=1000, means_init = means, weights_init = weights)
	gmm.fit(train_data)
	# print (np.shape(gmm.means_))
	# check the accuracy
	matched_pairs = 0
	i = 0
	for test in tests:
		min_distance = float('inf')
		best_template = ''
		s = len(test) # get the length
		test	
		S_scores = gmm.predict_proba(test)
		S_scores[S_scores == 0] = 0.0000001
		S_scores = np.log(S_scores)
		S_scores[S_scores == -inf] = -100000000
		# print (S_scores)
		j = 0
		for template in data:		
			t = len(template) # get the length 
			T_scores = gmm.predict_proba(template)
			T_scores[T_scores == 0] = 0.0000001
			T_scores = np.log(T_scores)
			T_scores[T_scores == -inf] = -100000000
			score_matrix_raw = [[-10 for x in range(t)] for y in range(s)]		
			score_matrix = np.array([np.array(xi) for xi in score_matrix_raw])
			# pass in s scores and t scores
			dist = dp(s-1,t-1,S_scores,T_scores,score_matrix)
			if dist < min_distance:
				min_distance = dist
				best_template = li_files[j]
			j = j+1

		test_file = li_tests[i]
		if best_template[0] is test_file[0]:
			matched_pairs +=1
		i = i+1
	result = float(matched_pairs)/11*100
	# round the number to 4th precision after the decimal points
	result = math.floor(result*10000)/10000
	str_result = str(result)+"%"
	print ("Accuracy using Mixture of Gaussians log Probability with {} classes is {} ".format(Gaussian_num,str_result))		
	return result
	# sample = samples[]
	# 

def main():
	# Try on the 32 Gaussians, default Gaussians
	num_Gaussians = 32
	# try on size_of_window = 512 
	window_size = 512
	# uniform distribution
	weights = [1/num_Gaussians for i in range(num_Gaussians)]

	num_files = len(li_files)
	samples = []
	for file in li_files:
		file_name = 'digits/'+file+'.wav'
		# print(file_name)
		(sf,S_array) = scipy.io.wavfile .read(file_name)
		mfcc,final_segements = MFCCEncoding(window_size,S_array,sf,frame_size = 0.01, shift = 0.005)
		# print (mfcc.shape[0])
		samples.append(mfcc)

	tests= []
	for file in li_tests:
		file_name = 'digits/'+file+'.wav'
		# print(file_name)
		(sf,S_array) = scipy.io.wavfile.read(file_name)
		mfcc,final_segements = MFCCEncoding(window_size,S_array,sf,frame_size = 0.01, shift = 0.005)
		# print (mfcc.shape[0])
		tests.append(mfcc)
	# store all MFCC samples into a list	
	# given the samples, weights is the initial weights
	print("--------------------------------------------------------------------------------")
	print("Default Results:")
	accuracy(Gaussian_num=num_Gaussians,weights=weights,means = None,data=samples,tests=tests)	
	print("--------------------------------------------------------------------------------")
	# combine weights initialization and means initilization 
	weights_list = []
	weights_list.append(weights)
	# random sample
	
	# use uniform distribution for initial weights for 3rd and 5th and use randomized distribution for 4th
	weights_list.append(weights)
	weights_4 = np.random.random_sample(num_Gaussians).tolist()
	weight_sum  = np.sum(weights_4)
	# normalize
	weights_list.append([weight/weight_sum for weight in weights_4])
	weights_list.append(weights)
	# print (samples)
	train_data = np.concatenate(samples, axis=0 )
	# take global mean of all data
	means = np.mean(train_data,axis = 0)
	# print (means)
	means_list = []
	means_list.append(means)
	means_new = 1.01 * means
	means_list.append(means_new)
	means_new_2 = 0.95 * means
	means_list.append(means_new_2)

	# print(means.shape[0])
	Gaussians_num_list = [k for k in range(1,8)]
	# Initialization = [] 

	fig, axes = plt.subplots(5,1,figsize=(10,8))
	for i in range(5):
		#create the weight array
		# 5 different initilization
		print("Initialization: ",i+1)
		weights_array = None
		accuracy_list = []
		for j in range(1,8):
			# create the repition of the means fo all classes
			num_Gaussians = 2**j # take from 2^1 to 2^7
			if i is 0 or i is 2:
				#random distribution on class coeffs
				weights_array = np.random.random_sample(num_Gaussians).tolist()
				weight_sum  = np.sum(weights_array)
				weights_array = np.array([weight/weight_sum for weight in weights_array])
			if i is 1 or i is 3:	
				# uniform distribution on class coeffs
				weights_array = np.array([1/num_Gaussians for i in range(num_Gaussians)])
			if i<=2:
				means = means_list[i]
				means = np.tile(means,(num_Gaussians,1))
			else:
				means = None
			accuracy_list.append(accuracy(Gaussian_num=num_Gaussians,weights=weights_array,means = means,data=samples,tests=tests))		
		
		# plt.subplot(3,2,i+1)
		# plt.figure
		# if i is 0:
		ax = axes[i]
		text ="Accuracy VS Gaussians Num in initilizations"
		if i is 0:
			ax.set_title(text)
		ax.plot(Gaussians_num_list,accuracy_list,'.-')
		ax.set_ylabel('Accuracy (%)')
		ax.set_xticks(Gaussians_num_list)
		ax.set_yticks([0,20,40,60,80,100])
		if i is 4:
			ax.set_xlabel('classes num (powers of 2)')	
		print("------------------------------------------------------------------------------")
		# plots the graphs
	fig.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)
	file_name = "plots_Gaussians"+".png"
	plt.savefig(file_name)
	plt.show()
	
	
main()




