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

def kalmanVel():
	directory = 'mp2_data_fixed.dat'
	data = getData(directory)
	[truthData,measData] = parseData(data)

	#constants
	F = np.matrix([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])  #constant vel
	H = np.matrix([[1,0,0,0],[0,1,0,0]])

	#initializers
	posEst = np.transpose(np.matrix([0,50,10,10]))
	cov = np.identity(4)  #just a guess
	q=.01
	Q = np.matrix([[q,0,0,0],[0,q,0,0],[0,0,q,0],[0,0,0,q]])
	r=100
	R = np.matrix([[r,0],[0,r]])

	for i in range(len(measData)):
		#state prediction
		statePred = F*posEst
		statePredCov = F*cov*np.transpose(F)+Q

		#measurement prediction
		measPred = H*statePred
		measPredCov = H*statePredCov*np.transpose(H)+R

		#associate the data together
		measurement = np.matrix(measData[i])
		measurement = np.transpose(measurement)
		innovation = measurement-measPred

		#update
		gain = statePredCov*np.transpose(H)*np.linalg.inv(measPredCov)
		posEst = statePred+gain*innovation
		cov = (np.identity(4)-gain*H)*statePredCov

		try:
			estimation = np.vstack((estimation,np.transpose(posEst)))
		except:
			estimation = np.transpose(posEst)

	plotData(truthData,measData,estimation,"Kalman Filter - Constant Velocity")

	return truthData,measData,estimation

def kalmanAccel():
	directory = 'mp2_data_fixed.dat'
	data = getData(directory)
	[truthData,measData] = parseData(data)

	#constants
	F = np.matrix([[1,0,1,0,1/2,0],[0,1,0,1,0,1/2],[0,0,1,0,1,0],[0,0,0,1,0,1],[0,0,0,0,1,0],[0,0,0,0,0,1]])  #constant accel
	H = np.matrix([[1,0,0,0,0,0],[0,1,0,0,0,0]])

	#initializers
	posEst = np.transpose(np.matrix([0,50,0,0,1,1]))
	cov = np.identity(6)  #just a guess
	q=0.1
	Q = np.matrix([[q,0,0,0,0,0],[0,q,0,0,0,0],[0,0,q,0,0,0],[0,0,0,q,0,0],[0,0,0,0,q,0],[0,0,0,0,0,q]])
	r=10000
	R = np.matrix([[r,0],[0,r]])

	for i in range(len(measData)):
		#state prediction
		statePred = F*posEst
		statePredCov = F*cov*np.transpose(F)+Q

		#measurement prediction
		measPred = H*statePred
		measPredCov = H*statePredCov*np.transpose(H)+R

		#associate the data together
		measurement = np.matrix(measData[i])
		measurement = np.transpose(measurement)
		innovation = measurement-measPred

		#update
		gain = statePredCov*np.transpose(H)*np.linalg.inv(measPredCov)
		posEst = statePred+gain*innovation
		cov = (np.identity(6)-gain*H)*statePredCov

		try:
			estimation = np.vstack((estimation,np.transpose(posEst)))
		except:
			estimation = np.transpose(posEst)

	plotData(truthData,measData,estimation,"Kalman Filter - Constant Acceleration")

	return truthData,measData,estimation


def plotData(truth,meas,est,name):
	plt.plot(truth[:,0],truth[:,1], label='truth')
	plt.scatter(meas[:,0], meas[:,1], label='measurement')
	plt.plot(est[:,0],est[:,1], label='estimate')
	plt.legend(loc=4)
	plt.axis('equal')
	plt.axis([0,350,0,350])
	plt.title(name)
	plt.xlabel('x position')
	plt.ylabel('y position')
	plt.show()

[truth,meas,est] = kalmanVel()
[truth,meas,est] = kalmanAccel()
# plotData(truth,meas,est)