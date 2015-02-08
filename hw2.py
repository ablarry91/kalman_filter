import numpy as np
import csv

# import data
	#filter by ground truth and noisy data

 #    20    20
 #    30    30
 #    40    40
 #    50    50
 #    60    60
 #    70    70
 #    80    80
 #    90    90
 #   100   100
 #   110   110
 #   120   120
 #   130   130
 #   140   140
 #   150   150
 #   160   160
 #   170   170
 #   180   180
 #   190   190
 #   200   200
 #   210   210
 #   220   220
 #   230   230
 #   240   240
 #   250   250
 #   260   260
 #   270   270
 #   280   280
 #   290   290
 #   300   300
 #   310   310

 # 30.6599   40.6570
 #    8.9595   37.2423
 #   39.2643   15.4473
 #   44.4980   46.7913
 #   38.3285   20.9157
 #   51.8103   69.8884
 #   45.5302  105.2615
 #   77.9913   48.7215
 #  102.2182  129.7523
 #  111.0600  113.2396
 #  119.4624  123.4715
 #  147.6434  133.6459
 #  155.1059  150.1607
 #  152.6376  155.6021
 #  140.3430  141.1183
 #  169.7388  177.0869
 #  162.1058  196.2422
 #  192.1907  244.6329
 #  208.2216  173.8628
 #  217.6761  219.9901
 #  209.7843  224.6984
 #  218.0435  230.4154
 #  248.3889  263.8221
 #  265.4243  197.1156
 #  265.7086  276.5219
 #  269.8376  287.1688
 #  295.4958  306.1189
 #  314.6301  309.1713
 #  266.9090  280.1921
 #  323.7047  290.5026

# def getData():
# 	truthData = []
# 	measData = []
# 	val = ''
# 	for line in open('mp2_data.txt'):
# 		if line[0:6] == 'Ground':
# 			print 'it worked'
# 		elif line[0:5] == 'Noisy':
# 			print 'this also worked'
# 		for i in line:
# 			if i == ' ' and len(val) >= 1:
# 				# print 'end of word detected'
# 				truthData.append(val)
# 				# print val
# 				val = ''
# 			elif i == ' ' and len(val) == 0:
# 				# print 'space.  do nothing'
# 				break
# 			val += i
# 	print truthData

def getData():
	pass


	# results = []
	# with open('mp2_data.txt') as inputfile:
	#     for row in csv.reader(inputfile):
	#         results.append(row)		
	# # print results
	pass

truthData = np.matrix([[20,20],[30,30],[40,40],[50,50],[60,60]])
measData = np.matrix([[30.6599, 40.6570],[8.9595, 37.2423],[39.1632,15.4473],[44.4980,46.7913],[38.3285,20.9157]])


# measData = np.matrix([[100, 40.6570],[97.9, 37.2423],[94.4,15.4473],[92.7,46.7913],[87.3,20.9157]])


#constants
F = np.matrix([[1,1],[0,1]])  #constant accel
G = np.matrix([0.5,1]) #constant accel
H = np.matrix([1,0]) #we're only measuring position
a = 1.0 #acceleration

#initializers
posEst = np.transpose(np.matrix([20,10]))
cov = np.zeros([2,2]) #initial estimate of the state covariance
cov = np.matrix([[.1,.1],[.1,.1]])  #just a guess
Q = np.matrix([[2,0],[0,2]])
R = np.matrix([1])

for i in range(len(measData)):

	#state prediction
	statePred = F*posEst+np.transpose(G)*a

	#measurement prediction
	measPred = H*statePred

	#measurement residual
	measRes = measData[i,0] - measPred

	#updated state estimate
		#update state covariance estimation
	statePredCov = F*cov*np.transpose(F)+Q
	measPredCov = H*statePredCov*np.transpose(H)+R
	gain = statePredCov*np.transpose(H)*np.linalg.inv(measPredCov)
	cov = statePredCov - gain*measPredCov*(np.linalg.pinv(gain))
	posEst = statePred + gain*measRes
	print 'statePred = ',statePred
	print 'measPred = ',measPred
	print 'measRes = ',measRes
	print 'statePredCov = ',statePredCov
	print 'measPredCov = ',measPredCov
	print 'gain = ',gain
	print 'cov = ',cov
	print 'posEst = ',posEst
	print ''

