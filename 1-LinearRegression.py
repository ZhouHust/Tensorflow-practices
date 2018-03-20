'''
a simple Linear Regression using tensorflow
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#create original data
xdata = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0, 0.05, xdata.shape)
ydata = xdata * 0.3 - 0.5 + noise

#create a model
Weights = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.zeros([1])) #random_uniform/... are ok
y = xdata * Weights + b

#optimization settings
loss = tf.reduce_mean(tf.square(ydata - y))  #Mean Square Error
optimizer = tf.train.GradientDescentOptimizer(0.1)   #Learning rate
train = optimizer.minimize(loss)

#initialization
init = tf.global_variables_initializer()

#create Session
with tf.Session() as sess:      #sess = tf.Session() & sess.close()
	sess.run(init)
	for step in range(500):
		sess.run(train)          #training
		if step % 50 == 0:
			print('loss :',sess.run(loss),'Weights :',sess.run(Weights),'bias :', sess.run(b))
	yp = sess.run(y)  #y_predict

#visualization	
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(xdata, ydata, marker ='.')
#plt.ion()
#plt.pause(1)    #pause for 1 second
ax.plot(xdata, yp, 'r-', lw=3)
plt.show()