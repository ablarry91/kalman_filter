import numpy as np
# import csv
from matplotlib import pyplot as plt

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

def getData(directory):
	data = np.loadtxt(directory)
	return data

def parseData(data):
	pos = data[:,0:2]
	meas = data[:,2:4]
	return pos, meas

# MINI SAMPLE OF THE HW DATA
# truthData = np.matrix([[20,20],[30,30],[40,40],[50,50],[60,60]])
# measData = np.matrix([[30.6599, 40.6570],[8.9595, 37.2423],[39.1632,15.4473],[44.4980,46.7913],[38.3285,20.9157]])

# MINI EXAMPLE OF A BALL FALLING IN GRAVITY
# measData = np.matrix([[100, 40.6570],[97.9, 37.2423],[94.4,15.4],[92.7,46.7913],[87.3,20.9157]])
# truthData = np.matrix([[100,5],[99.5,5],[98,5],[95.5,5],[92,5],[87.5,5]])

def kalman():
	directory = 'mp2_data_fixed.dat'
	data = getData(directory)
	[truthData,measData] = parseData(data)
	# truthData = truthData[:,0]
	# measData = measData[:,0]

	#constants
	F = np.matrix([[1,1],[0,1]])  #constant accel
	G = np.matrix([0.5,1]) #constant accel
	H = np.matrix([1,0]) #we're only measuring position
	a = .10 #acceleration

	#initializers
	posEst = np.transpose(np.matrix([20,10]))
	# cov = np.zeros([2,2]) #initial estimate of the state covariance
	cov = np.matrix([[1,1],[1,1]])  #just a guess
	Q = np.matrix([[1,0],[0,1]])
	R = np.matrix([1])

	# count = 0
	# estimation = np.array([0])
	for i in range(len(measData)):

		#state prediction
		statePred = F*posEst+np.transpose(G)*a
		#measurement prediction
		measPred = H*statePred

		#measurement residual
		try:
			measRes = measData[i,0] - measPred
		except:
			break

		#updated state estimate
			#update state covariance estimation
		statePredCov = F*cov*np.transpose(F)+Q
		measPredCov = H*statePredCov*np.transpose(H)+R
		gain = statePredCov*np.transpose(H)*np.linalg.inv(measPredCov)
		cov = statePredCov - gain*measPredCov*(np.transpose(gain))
		# cov = (np.identity(2)-gain*H)*statePredCov
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
		try:
			estimation = np.vstack((estimation,posEst[0]))
		except:
			estimation = posEst[0]
	plotData(truthData,measData,estimation)

def plotData(truth,meas,est):
	plt.plot(truth[:,0], label='truth')
	plt.plot(meas[:,0], label='measurement')
	plt.plot(est, label='estimate')
	plt.legend()
	plt.show()

