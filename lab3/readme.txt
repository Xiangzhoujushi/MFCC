Required part Code:
MFCC.py
dtw.py
Extension 3:
MOG.py

Intermediate file for debugging:
MOG.ipynb
MFCC.ipynb

Files:
1. MFCC.py: the file that
2. dtw.py : the file that implements the dynamic warping algorithms, and the main methods in comparing the log spec and MFCC algorithms
3. MOG.py: the file implements mixture of Gaussians.

The findings on the required part and extension part

Required (c) part: 
Log spec representations have higher distance difference than that of MFCC representations
We have 100% of accuracy for both representations, there are no mismatched pairs.

For the extension part:
We see the overfitting issue of using too many Gaussians if we are not using the best initializations. The accuracy rate first increase, then might drop, if the initialization is not stable. And it is also because we do not have much data. (Because we only have around 30 phones, the use of 64 or 128 classes would cause overfitting issues, and the variation rate depends the stability of our chosen initiation)

Run the file:
Commands:
Required part: python3 dtw.py 1a 1b
1a and 1b can be replaced by any other files

Extension part: python3 MOG.py
Please wait for 3 minutes to get the plots and all the results in the terminal
1. Init: Uniform weights, with global Mean
2. Init: Random weights, with global Mean*(1.01)
3. Init: Uniform ..., with global Mean*(0.97)
4. Init: Random weights, default k-means
5. Init: Best Randomized weights, default k-means

The results are saved to the plots_Gaussians.png. (you can take a look of it)

