'''
curve fitting , simple Neural Networks, tensorflow
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# define a function for hidden layers
def add_layer(inputs, in_size, out_size, activation_function=None):
	Weights = tf.Variable(tf.random_normal([in_size,out_size]))
	biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
	z_W_plus_b = tf.matmul(inputs, Weights) + biases     #z in g(z),refers to (x * theta)
	if activation_function is None:     # case: equal to Linear Regression
		outputs = z_W_plus_b
	else:
		outputs = activation_function(z_W_plus_b)
	return outputs

#create original data
xdata = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0, 0.05, xdata.shape)
ydata = np.square(xdata) + noise +0.5

xs = tf.placeholder(tf.float32,[None,1])   # formal parameter,used to define the process and 
ys = tf.placeholder(tf.float32,[None,1])   # assign a specific value at the time of execution.

h_layer1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)  #hidden layer
y_pred = add_layer(h_layer1, 10, 1,activation_function=tf.nn.relu)  #final and output layer

loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_pred - ys),reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(0.3)
train = optimizer.minimize(loss)

# visualize original data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(xdata, ydata)
plt.ion()
plt.show()
plt.pause(1)

#trainning
with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	for step in range(1000):
		sess.run(train,feed_dict={xs:xdata,ys:ydata})
		## pair with placeholder,if placeholder is used in your computation process;
		## you should put feed_dict in there.
		if step % 100 == 0:
			print ('loss:',sess.run(loss,feed_dict={xs:xdata,ys:ydata}))
			try:
				ax.lines.remove(lines[0])
			except Exception:
				pass
			y_pred_value = sess.run(y_pred,feed_dict={xs:xdata,ys:ydata})
			lines = ax.plot(xdata, y_pred_value,'r-',lw=5)  #visualize the fitting curve
			plt.pause(1)