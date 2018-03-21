'''
classification , mnist, tensorflow
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
# data: number 1 to 10
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

def add_layer(inputs, in_size, out_size, activation_function=None):
	Weights = tf.Variable(tf.random_normal([in_size,out_size]))
	biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
	z_W_plus_b = tf.add(tf.matmul(inputs, Weights),biases)    #z in g(z),refers to (x * theta)
	if activation_function is None:     # case: equal to Linear Regression
		outputs = z_W_plus_b
	else:
		outputs = activation_function(z_W_plus_b)
	return outputs

def compute_accuracy(v_xs,v_ys):
	#global y_pred
	y_pred_value = sess.run(y_pred,feed_dict={xs:v_xs})
	correct_prediction = tf.equal(tf.argmax(y_pred_value,1),tf.argmax(v_ys,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
	return result

#define placeholder for inputs to network
xs = tf.placeholder(tf.float32,[None,784])   # 28x28
ys = tf.placeholder(tf.float32,[None,10])   # 10 classes

#h_layer1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)  #hidden layer
y_pred = add_layer(xs, 784, 10,activation_function=tf.nn.softmax)  #final and output layer

# the error between y_pred and real data
loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(y_pred),reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

sess =tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

loss_list = []
loss_time = []
accu_list = []
for i in range(1000):
	batch_xs,batch_ys = mnist.train.next_batch(100)    #Stochastic Gradient Descent
	sess.run(train,feed_dict={xs:batch_xs,ys:batch_ys})
	if i % 50 == 0:
		loss_list.append(sess.run(loss,feed_dict={xs:batch_xs,ys:batch_ys}))
		loss_time.append(i//50)
		accu_list.append(compute_accuracy(mnist.test.images,mnist.test.labels))
		print(compute_accuracy(mnist.test.images,mnist.test.labels))

np_loss = np.array(loss_list)
np_time = np.array(loss_time)
np_accu = np.array(accu_list)

fig1 = plt.figure()
ax = fig1.add_subplot(1,1,1)
ax.plot(np_time,np_loss,'-b')
ax.set_xlabel('times/50')
ax.set_ylabel('loss')
ax.set_title('loss-time curve')

fig2 = plt.figure()
bx = fig2.add_subplot(1,1,1)
bx.plot(np_time,np_accu,'-r')
bx.set_xlabel('times/50')
bx.set_ylabel('accuracy')
bx.set_title('accuracy-time curve')
plt.show()