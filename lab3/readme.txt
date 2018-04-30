Required part Code:
MFCC.py
dtw.py (23 points)
Extension 3: (10 points)
MOG.py

Extension 2: (2 points)
dtw.py

Original wav file and students' record files:
digits1 folder

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
(Please type Y or N (capitalized) to see the results of c and d parts for interactive questions)

(Please type Y or N (capitalized) to see the results of c and d parts for interactive questions)
(For extension 1 as well, it takes some time to run the extension, as the file is large, we use two students' voice of our team) The accuracy drops as there are noises when we record the voices (because my English pronunciation will affect the result. LOL)(Our accuracy rate using student's file are around 40%) and it takes time to run
 
Extension 1:
Accuracy: 
42.424%% for MFCC, 45.2454% for log  (students as test cases)
9.09% for MFCC and 9.09% for log using (student 1's files as templates)
21.21% for MFCC and 21.21% for log using (student 2 as's files the templates)

(The reason is because our files are 10 times longer than the templates, and other parts are filled with silence and noise), only 1/10 portions are useful data

Extension 3: python3 MOG.py
Please wait for 3 minutes to get the plots and all the results in the terminal
1. Init: Uniform weights, with global Mean
2. Init: Random weights, with global Mean*(1.01)
3. Init: Uniform ..., with global Mean*(0.97)
4. Init: Random weights, default k-means
5. Init: Best Randomized weights, default k-means

The results are saved to the plots_Gaussians.png. (you can take a look of it)


