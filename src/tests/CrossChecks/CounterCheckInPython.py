import math 
import numpy as np
 
def sigmoid(x):
    return 1/(1+math.exp(-x))
    
def sigmoid_prime(sigmoid_value):
    return sigmoid_value*(1-sigmoid_value)  
    
sigmoid_matrix = np.frompyfunc(sigmoid, 1, 1)    
sigmoid_prime_matrix = np.frompyfunc(sigmoid_prime, 1, 1)
    
s0 = np.matrix([[0.18, 0.29, 0.40, 0.51, 0.62]])
w0 = np.matrix([[0.01, 0.02, 0.03, 0.04],
                [0.05, 0.06, 0.07, 0.08],
                [0.09, 0.10, 0.11, 0.12],
                [0.13, 0.14, 0.15, 0.16],
                [0.17, 0.18, 0.19, 0.20]])
b0 = np.matrix([ 0.21, 0.22, 0.23, 0.24 ])
w1 = np.matrix([[ 0.25, 0.26, 0.27 ],
			[ 0.28, 0.29, 0.30 ],
			[ 0.31, 0.32, 0.33 ],
			[ 0.34, 0.35, 0.36 ]])
b1 = np.matrix([0.37, 0.38, 0.39])
w2 = np.matrix([[ 0.40, 0.41 ],
			[ 0.42, 0.43 ],
			[ 0.44, 0.45 ]])
b2 = np.matrix([ 0.46, 0.47])

#forward
s1 = sigmoid_matrix(s0*w0+b0)
s2 = sigmoid_matrix(s1*w1+b1)
s3 = sigmoid_matrix(s2*w2+b2)

print 'forward output =', s3

#backward
speed = 0.5
expected = np.matrix('0.01 0.99')

#Find b2_update
dETotal_dS3 = s3 - expected
dS3_dN2 = sigmoid_prime_matrix(s3)
dN2_dB2 = np.ones((1, 2))
dETotal_dB2 = np.multiply(dETotal_dS3, dS3_dN2, dN2_dB2)
b2_update = dETotal_dB2 * speed
print 'b2_update =', b2_update

#Find w2_update
dETotal_dS3 = s3 - expected
dS3_dN2 = sigmoid_prime_matrix(s3)
dETotal_dN2 = np.multiply(dETotal_dS3, dS3_dN2)
dN2_dW2 = np.concatenate((s2, s2), 0).transpose()
dETotal_dW2 = np.matrix(np.zeros(dN2_dW2.shape))
dETotal_dW2[:,0] = np.multiply(dN2_dW2[:,0], dETotal_dN2[:,0])
dETotal_dW2[:,1] = np.multiply(dN2_dW2[:,1], dETotal_dN2[:,1])
w2_update = dETotal_dW2 * speed
print 'w2_update =', w2_update

#Find b1_update
dETotal_dS3 = s3 - expected
dS3_dN2 = sigmoid_prime_matrix(s3)
dETotal_dN2 = np.multiply(dETotal_dS3, dS3_dN2)
dN2_dS2 = w2.transpose()
dETotal_dS2 = np.matrix(np.zeros(s2.shape))
dETotal_dS2 = dETotal_dS2 + np.multiply(dN2_dS2[0, :], dETotal_dN2[:, 0])
dETotal_dS2 = dETotal_dS2 + np.multiply(dN2_dS2[1, :], dETotal_dN2[:, 1])
dS2_dN1 = sigmoid_prime_matrix(s2)
dN1_dB1 = np.ones(b1.shape)
dETotal_dB1 = np.multiply(dETotal_dS2, dS2_dN1, dN1_dB1)
b1_update = dETotal_dB1 * speed
print 'b1_update =', b1_update

#Find w1_update
dETotal_dS3 = s3 - expected
dS3_dN2 = sigmoid_prime_matrix(s3)
dETotal_dN2 = np.multiply(dETotal_dS3, dS3_dN2)
dN2_dS2 = w2.transpose()
dETotal_dS2 = np.matrix(np.zeros(s2.shape))
dETotal_dS2 = dETotal_dS2 + np.multiply(dN2_dS2[0, :], dETotal_dN2[:, 0])
dETotal_dS2 = dETotal_dS2 + np.multiply(dN2_dS2[1, :], dETotal_dN2[:, 1])
dS2_dN1 = sigmoid_prime_matrix(s2)
dETotal_dN1 = np.multiply(dETotal_dS2, dS2_dN1)
dN1_dW1 = np.concatenate((s1, s1, s1), 0).transpose()
dETotal_dW1 = np.matrix(np.zeros(dN1_dW1.shape))
dETotal_dW1[:,0] = np.multiply(dN1_dW1[:,0], dETotal_dN1[:,0])
dETotal_dW1[:,1] = np.multiply(dN1_dW1[:,1], dETotal_dN1[:,1])
dETotal_dW1[:,2] = np.multiply(dN1_dW1[:,2], dETotal_dN1[:,2])
w1_update = dETotal_dW1 * speed
print 'w1_update =', w1_update

#find b0_update
dETotal_dS3 = s3 - expected
dS3_dN2 = sigmoid_prime_matrix(s3)
dETotal_dN2 = np.multiply(dETotal_dS3, dS3_dN2)
dN2_dS2 = w2.transpose()
dETotal_dS2 = np.matrix(np.zeros(s2.shape))
dETotal_dS2 = dETotal_dS2 + np.multiply(dN2_dS2[0, :], dETotal_dN2[:, 0])
dETotal_dS2 = dETotal_dS2 + np.multiply(dN2_dS2[1, :], dETotal_dN2[:, 1])
dS2_dN1 = sigmoid_prime_matrix(s2)
dETotal_dN1 = np.multiply(dETotal_dS2, dS2_dN1)
dN1_dS1 = w1.transpose()
dETotal_dS1 = np.matrix(np.zeros(s1.shape))
dETotal_dS1 = dETotal_dS1 + np.multiply(dN1_dS1[0, :], dETotal_dN1[:, 0])
dETotal_dS1 = dETotal_dS1 + np.multiply(dN1_dS1[1, :], dETotal_dN1[:, 1])
dETotal_dS1 = dETotal_dS1 + np.multiply(dN1_dS1[2, :], dETotal_dN1[:, 2])
dS1_dN0 = sigmoid_prime_matrix(s1)
dN0_dB0 = np.ones(b0.shape)
dETotal_dB0 = np.multiply(dETotal_dS1, dS1_dN0, dN0_dB0)
b0_update = dETotal_dB0 * speed
print 'b0_update =', b0_update

#find w0_update
dETotal_dS3 = s3 - expected
dS3_dN2 = sigmoid_prime_matrix(s3)
dETotal_dN2 = np.multiply(dETotal_dS3, dS3_dN2)
dN2_dS2 = w2.transpose()
dETotal_dS2 = np.matrix(np.zeros(s2.shape))
dETotal_dS2 = dETotal_dS2 + np.multiply(dN2_dS2[0, :], dETotal_dN2[:, 0])
dETotal_dS2 = dETotal_dS2 + np.multiply(dN2_dS2[1, :], dETotal_dN2[:, 1])
dS2_dN1 = sigmoid_prime_matrix(s2)
dETotal_dN1 = np.multiply(dETotal_dS2, dS2_dN1)
dN1_dS1 = w1.transpose()
dETotal_dS1 = np.matrix(np.zeros(s1.shape))
dETotal_dS1 = dETotal_dS1 + np.multiply(dN1_dS1[0, :], dETotal_dN1[:, 0])
dETotal_dS1 = dETotal_dS1 + np.multiply(dN1_dS1[1, :], dETotal_dN1[:, 1])
dETotal_dS1 = dETotal_dS1 + np.multiply(dN1_dS1[2, :], dETotal_dN1[:, 2])
dS1_dN0 = sigmoid_prime_matrix(s1)
dETotal_dN0 = np.multiply(dETotal_dS1, dS1_dN0)
dN0_dW0 = np.concatenate((s0, s0, s0, s0), 0).transpose()
dETotal_dW0 = np.matrix(np.zeros(dN0_dW0.shape))
dETotal_dW0[:,0] = np.multiply(dN0_dW0[:,0], dETotal_dN0[:,0])
dETotal_dW0[:,1] = np.multiply(dN0_dW0[:,1], dETotal_dN0[:,1])
dETotal_dW0[:,2] = np.multiply(dN0_dW0[:,2], dETotal_dN0[:,2])
dETotal_dW0[:,3] = np.multiply(dN0_dW0[:,3], dETotal_dN0[:,3])
w0_update = dETotal_dW0 * speed
print 'w0_update =', w0_update

#Actual updates
b2 = b2 - b2_update
print 'b2 =', b2
w2 = w2 - w2_update
print 'w2 =', w2
b1 = b1 - b1_update
print 'b1 =', b1
w1 = w1 - w1_update
print 'w1 =', w1
b0 = b0 - b0_update
print 'b0 =', b0
w0 = w0 - w0_update
print 'w0 =', w0
