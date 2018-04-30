#Peiyuan's version of MFCC algorithm,

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.io.wavfile 
import math
import sys
from scipy.fftpack import dct
from MFCC import MFCCEncoding

# set the limit of recursion
sys.setrecursionlimit(150000)


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

# display the accuray and other 


first_file = 'digits1/'+file1+'.wav'
second_file = 'digits1/'+file2+'.wav'
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

MFCC_S,log_S = MFCCEncoding(window_size = 512,signal = S_array,sample_rate = sf1,frame_size = 0.025, shift = 0.01)
MFCC_T,log_T = MFCCEncoding(window_size = 512,signal = T_array,sample_rate = sf2,frame_size = 0.025, shift = 0.01)

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

# the templates list and the tests list in our model
li_templates = ['1a','2a','3a','4a','5a','6a','7a','8a','9a','za','oa']
li_tests = ['1b','2b','3b','4b','5b','6b','7b','8b','9b','zb','ob']
li_student2 = ['1','2','3','4','5','6','7','8','9','z','o']
li_student1 = [string+'patrick' for string in li_student2]
li_tests_all = (li_tests+li_student2+li_student1)
# print (li_tests_all)
li_tests_against_1 = li_templates+li_tests+li_student2
li_tests_against_2 = li_templates+li_tests+li_student1

# waveform = dict()


# use the minimum distance file as the right label, log representation
ans = input("Do you want to see the result of accuracy of two distances between all test and template pairs (Y/N): ")

if ans is 'Y':
	matched_pairs_log  = 0 # mathced pairs using log representation
	matched_pairs_MFCC = 0 # matched pairs using MFCC representation
	for test in li_tests:
		test_file = 'digits1/'+test+'.wav'
		min_distance_log = float('inf')
		min_distance_MFCC = float('inf')
		temp_log = ""
		temp_MFCC = ""
		(sf1,S_array) = scipy.io.wavfile .read(test_file)
		MFCC_S,log_S = MFCCEncoding(window_size = 512,signal = S_array,sample_rate = sf1,frame_size = 0.025, shift = 0.01)
		s = len(MFCC_S) # get the length 
		for template in li_templates:
			template_file= 'digits1/'+template+'.wav'	
			(sf2,T_array) = scipy.io.wavfile .read(template_file)
			MFCC_T,log_T = MFCCEncoding(window_size = 512,signal = T_array,sample_rate = sf2,frame_size = 0.025, shift = 0.01)
			t = len(MFCC_T) # get the length
			score_matrix_raw = [[-10 for x in range(t)] for y in range(s)]		
			score_matrix = np.array([np.array(xi) for xi in score_matrix_raw]) # score matrix for MFCC
			score_matrix_1 = score_matrix.copy()
			# set the min distance template
			if dp(s-1,t-1,MFCC_S,MFCC_T)<min_distance_MFCC:
				min_distance_MFCC = dp(s-1,t-1,MFCC_S,MFCC_T)
				temp_MFCC = template
			# set the min distance template
			score_matrix = score_matrix_1
			if dp(s-1,t-1,log_S,log_T)<min_distance_log:
				min_distance_log = dp(s-1,t-1,log_S,log_T)
				temp_log = template

		if temp_log[0] is test[0]:
			matched_pairs_log +=1
		if temp_MFCC[0] is test[0]:
			matched_pairs_MFCC +=1
		# 
	print ("Accuracy using MFCC distance is: ",str(float(matched_pairs_MFCC)/11*100)+'%')
	print ("Accuracy using log spec distance is: ",str(float(matched_pairs_log)/11*100)+'%')

#extension 1

ans = input("Do you want to see the results of extension1 using stundents's files as test cases (Y/N): ")
if ans is 'Y':
	print("Number of test cases: ", len(li_tests_all))
	# use stundet's digits as all test cases
	matched_pairs_log  = 0 # mathced pairs using log representation
	matched_pairs_MFCC = 0 # matched pairs using MFCC representation
	for test in li_tests_all:
		# print (test)
		test_file = 'digits1/'+test+'.wav'
		min_distance_log = float('inf')
		min_distance_MFCC = float('inf')
		temp_log = ""
		temp_MFCC = ""
		(sf1,S_array) = scipy.io.wavfile .read(test_file)
		# choose the left audio recording if using double side recording
		if S_array.ndim == 2:
			S_array = S_array[:,0]
		MFCC_S,log_S = MFCCEncoding(window_size = 512,signal = S_array,sample_rate = sf1,frame_size = 0.025, shift = 0.01)
		s = len(MFCC_S) # get the length 
		for template in li_templates:
			template_file= 'digits1/'+template+'.wav'		
			(sf2,T_array) = scipy.io.wavfile .read(template_file)
			# choose the right audio recording if using double side recording
			if T_array.ndim == 2:
				T_array = T_array[:,0]
			# print("test shapes: ",np.shape(S_array))
			# print("template shapes: ",np.shape(T_array))
			MFCC_T,log_T = MFCCEncoding(window_size = 512,signal = T_array,sample_rate = sf2,frame_size = 0.025, shift = 0.01)
			t = len(MFCC_T) # get the length
			score_matrix_raw = [[-10 for x in range(t)] for y in range(s)]		
			score_matrix = np.array([np.array(xi) for xi in score_matrix_raw]) # score matrix for MFCC
			score_matrix_1 = score_matrix.copy()
			# set the min distance template
			if dp(s-1,t-1,MFCC_S,MFCC_T)<min_distance_MFCC:
				min_distance_MFCC = dp(s-1,t-1,MFCC_S,MFCC_T)
				temp_MFCC = template
			# set the min distance template
			score_matrix = score_matrix_1
			if dp(s-1,t-1,log_S,log_T)<min_distance_log:
				min_distance_log = dp(s-1,t-1,log_S,log_T)
				temp_log = template

		if temp_log[0] is test[0]:
			matched_pairs_log +=1
		if temp_MFCC[0] is test[0]:
			matched_pairs_MFCC +=1
	print ("Accuracy using MFCC distance on students' test cases is: ",str(float(matched_pairs_MFCC)/len(li_tests_all)*100)+'%')
	print ("Accuracy using log spec on students' test cases distance is: ",str(float(matched_pairs_log)/len(li_tests_all)*100)+'%')

ans = input("Do you want to use student 1  as the template (Y/N): ")
if ans is 'Y':
	print("Using Student 1 as templates------------------------------------")
	print("Number of test cases: ", len(li_tests_against_1))
	# #use student 1's recorded wav as templates
	# #use stundet's digits as all test cases
	matched_pairs_log  = 0 # mathced pairs using log representation
	matched_pairs_MFCC = 0 # matched pairs using MFCC representation
	for test in li_tests_against_1:
		test_file = 'digits1/'+test+'.wav'
		min_distance_log = float('inf')
		min_distance_MFCC = float('inf')
		temp_log = ""
		temp_MFCC = ""
		if S_array.ndim == 2:
			S_array = S_array[:,0]
		# S_array = np.ravel(S_array)
		MFCC_S,log_S = MFCCEncoding(window_size = 512,signal = S_array,sample_rate = sf1,frame_size = 0.025, shift = 0.01)
		s = len(MFCC_S) # get the length 
		for template in li_student1:
			template_file= 'digits1/'+template+'.wav'
			(sf2,T_array) = scipy.io.wavfile .read(template_file)
			if T_array.ndim == 2:
				T_array = T_array[:,0]
			MFCC_T,log_T = MFCCEncoding(window_size = 512,signal = T_array,sample_rate = sf2,frame_size = 0.025, shift = 0.01)
			t = len(MFCC_T) # get the length
			score_matrix_raw = [[-10 for x in range(t)] for y in range(s)]		
			score_matrix = np.array([np.array(xi) for xi in score_matrix_raw]) # score matrix for MFCC
			score_matrix_1 = score_matrix.copy()
			# set the min distance template
			if dp(s-1,t-1,MFCC_S,MFCC_T)<min_distance_MFCC:
				min_distance_MFCC = dp(s-1,t-1,MFCC_S,MFCC_T)
				temp_MFCC = template
			# set the min distance template
			score_matrix = score_matrix_1
			if dp(s-1,t-1,log_S,log_T)<min_distance_log:
				min_distance_log = dp(s-1,t-1,log_S,log_T)
				temp_log = template

		if temp_log[0] is test[0]:
			matched_pairs_log +=1
		if temp_MFCC[0] is test[0]:
			matched_pairs_MFCC +=1
	print ("Accuracy using MFCC distance and student 1's files as templates is: ",str(float(matched_pairs_MFCC)/33*100)+'%')
	print ("Accuracy using log spec distance and student 1's files as templates is: ",str(float(matched_pairs_log)/33*100)+'%')

# matched_pairs_log  = 0 # mathced pairs using log representation
# matched_pairs_MFCC = 0 # matched pairs using MFCC representation
ans = input("Do you want to use student 2 as the template (Y/N): ")
if ans is 'Y':
	print("Using Student 2 as templates------------------------------------")
	print("Number of test cases: ", len(li_tests_against_2))
	for test in li_tests_against_2:
		test_file = 'digits1/'+test+'.wav'
		min_distance_log = float('inf')
		min_distance_MFCC = float('inf')
		temp_log = ""
		temp_MFCC = ""
		(sf1,S_array) = scipy.io.wavfile .read(test_file)
		if S_array.ndim == 2:
			S_array = S_array[:,0]
		MFCC_S,log_S = MFCCEncoding(window_size = 512,signal = S_array,sample_rate = sf1,frame_size = 0.025, shift = 0.01)
		s = len(MFCC_S) # get the length 
		for template in li_student2:
			template_file= 'digits1/'+template+'.wav'
			(sf2,T_array) = scipy.io.wavfile .read(template_file)
			if T_array.ndim == 2:
				T_array = T_array[:,0]
			MFCC_T,log_T = MFCCEncoding(window_size = 512,signal = T_array,sample_rate = sf2,frame_size = 0.025, shift = 0.01)		
			t = len(MFCC_T) # get the length
			score_matrix_raw = [[-10 for x in range(t)] for y in range(s)]		
			score_matrix = np.array([np.array(xi) for xi in score_matrix_raw]) # score matrix for MFCC
			score_matrix_1 = score_matrix.copy()
			# set the min distance template
			if dp(s-1,t-1,MFCC_S,MFCC_T)<min_distance_MFCC:
				min_distance_MFCC = dp(s-1,t-1,MFCC_S,MFCC_T)
				temp_MFCC = template
			# set the min distance template
			score_matrix = score_matrix_1
			if dp(s-1,t-1,log_S,log_T)<min_distance_log:
				min_distance_log = dp(s-1,t-1,log_S,log_T)
				temp_log = template

		if temp_log[0] is test[0]:
			matched_pairs_log +=1
		if temp_MFCC[0] is test[0]:
			matched_pairs_MFCC +=1
	#use stundet 2's recorded wav as templates
	print ("Accuracy using MFCC distance and student 2's files as templates is: ",str(float(matched_pairs_MFCC)/33*100)+'%')
	print ("Accuracy using log spec distance and student 2's files as templates is: ",str(float(matched_pairs_log)/33*100)+'%')
	
	








