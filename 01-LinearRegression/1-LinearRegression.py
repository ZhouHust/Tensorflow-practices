'''
Linear Regression -tensorflow
& 3D data visualization
--ZhouHust
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

# generate data. 100 points total
x_data = np.float32(np.random.rand(2, 100))
y_data = np.dot([0.1, 0.2], x_data) + 0.3  #np.dot matrix multiply

# create a Linear model
b = tf.Variable(tf.zeros([1]))  # bias
w = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(w, x_data) + b

# minimize the cost function. Mean-Square-error
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# initialization
init = tf.global_variables_initializer()

# Start graph
sess = tf.Session()
sess.run(init)

# fitting
for step in range(0,201):
	sess.run(train)
	if step % 20 == 0 :
		print(step, sess.run(w), sess.run(b))

# visualize the original data
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(x_data[0,:], x_data[1,:], y_data[:],c='b',marker='.',s=50) # s decide the size of symbol
ax1.set_xlabel('x1 value',labelpad=5,fontsize=15)
ax1.set_ylabel('x2 value',labelpad=5,fontsize=15)
ax1.set_zlabel('y value',labelpad=5,fontsize=15)
ax1.set_title('data visualization',fontsize=15)

# visualize the result
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.scatter(x_data[0,:], x_data[1,:], y_data[:],c='b',marker='.',s=50) # s decide the size of symbol
ax2.set_xlabel('x1 value',labelpad=5,fontsize=15)
ax2.set_ylabel('x2 value',labelpad=5,fontsize=15)
ax2.set_zlabel('y value',labelpad=5,fontsize=15)
ax2.set_title('data visualization',fontsize=15)

x1_line = np.arange(0,1,0.02)
x2_line = np.arange(0,1,0.02)
X, Y = np.meshgrid(x1_line, x2_line)
Z = np.dot(sess.run(w), [x1_line,x2_line]) + sess.run(b)

ax2.plot_surface(X, Y, Z)
plt.show()
