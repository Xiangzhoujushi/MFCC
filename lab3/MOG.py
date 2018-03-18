#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
# import matplotlib.pyplot as plt
import scipy.io.wavfile 
import math
from scipy.fftpack import dct
from MFCC import MFCCEncoding



def main():
	li_files = ['1a','2a','3a','4a','5a','6a','7a','8a','9a','za','oa']
	window_size = 256
	num_files = len(li_files)
	samples = []
	for file in li_files:
		file_name = 'digits/'+file+'.wav'
		(sf,S_array) = scipy.io.wavfile .read(file_name)
		mfcc,final_segements = MFCCEncoding(window_size,S_array,sf)
		samples.append(mfcc)
	# store all MFCC samples into a list

	print(len(samples))

main()




