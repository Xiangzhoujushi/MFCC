#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
# import matplotlib.pyplot as plt
import scipy.io.wavfile 
import math
from scipy.fftpack import dct
from MFCC import MFCCEncoding
from sklearn import mixture  
from numpy import inf

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

def accuracy(Gaussian_num,weights,data,tests):
	
	# combine the array of the data into a single array
	train_data = np.concatenate(data, axis=0 )
	# weights = [1/num_Gaussians for i in range(num_Gaussians)]
	# print (weights)
	# choose 32 for the first time, use diagonal convariance type, uniform piority for each gaussian components
	gmm = mixture.GaussianMixture(n_components = Gaussian_num,covariance_type = 'diag',max_iter=1000, weights_init = weights)
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
	result = str(float(matched_pairs)/11*100)+'%'
	print ("Accuracy using Mixture of Gaussians log Probability with {} classes and default initialization is {}: ".format(Gaussian_num,result))		

	# sample = samples[]
	# 

def main():
	# Try on the 32 Gaussians
	num_Gaussians = 60
	# try on size_of_window = 512 
	window_size = 512
	# train our model using the sampls
	weights = [1/num_Gaussians for i in range(num_Gaussians)]
	
	num_files = len(li_files)
	samples = []
	for file in li_files:
		file_name = 'digits/'+file+'.wav'
		# print(file_name)
		(sf,S_array) = scipy.io.wavfile .read(file_name)
		mfcc,final_segements = MFCCEncoding(window_size,S_array,sf)
		# print (mfcc.shape[0])
		samples.append(mfcc)

	tests= []
	for file in li_tests:
		file_name = 'digits/'+file+'.wav'
		# print(file_name)
		(sf,S_array) = scipy.io.wavfile.read(file_name)
		mfcc,final_segements = MFCCEncoding(window_size,S_array,sf)
		# print (mfcc.shape[0])
		tests.append(mfcc)
	# store all MFCC samples into a list	
	# given the samples, weights is the initial weights
	accuracy(num_Gaussians,weights,samples,tests)	

main()




