#Peiyuan's version of MFCC algorithm,

import numpy as np

# wave form part 1
# wave form part 2
S = np.array([1,2,3,4]).tolist()
T = np.array([1,2,2,4,2,3,3]).tolist()
	# size of s

s = len(S)

	# size of t
t = len(T)

	# array
	# x
	# array
	# y
	# initiate all matrix to be from
score_matrix = [[-10 for x in range(t)] for y in range(s)]

#The function that return the minimum distance, given two position
def dp(i,j):
	if score_matrix[i][j]>0:
		return score_matrix[i][j]

	elif i is 0 and j is 0:
		score_matrix[i][j] = dist(0,0)
		return score_matrix[i][j]

	elif i is 0:
		score_matrix[i][j] = min(dp(i,j-1), dp(i,j-1))+dist(i,j)
		return score_matrix[i][j]

	elif j is 0:
		score_matrix[i][j] =  dp(i-1,j)+dist(i,j)
		return score_matrix[i][j]

	else:
		score_matrix[i][j]= min(dp(i-1,j), dp(i-1,j-1), dp(i,j-1))+dist(i,j)
		return score_matrix[i][j]

def dist(i,j):
	return abs(S[i]-T[j])
# import 
def main():
	# dist_matrix = 
	print ("Minimum xDistance: ",dp(s-1,t-1))

main()






