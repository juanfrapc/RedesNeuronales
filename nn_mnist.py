import gzip
import cPickle

import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h

def cuenta_errores(batch_xt, batch_yt):
	contador=0
	result = sess.run(y, feed_dict={x: batch_xt})
	for b, r in zip(batch_yt, result):
	    if np.argmax(b) != np.argmax(r):
		contador += 1;
	return contador

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set


# ---------------- Visualizing some element of the MNIST dataset --------------

#
#import matplotlib.cm as cm
#import matplotlib.pyplot as plt

#plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
#plt.show()  # Let's see a sample
#print train_y[57]

y_data_train = one_hot(train_y[:].astype(int), 10)  # the labels are in the last row. Then we encode them in one hot code
y_data_valid = one_hot(valid_y[:].astype(int), 10)  # the labels are in the last row. Then we encode them in one hot code
y_data_test = one_hot(test_y[:].astype(int), 10)  # the labels are in the last row. Then we encode them in one hot code

print train_x[57].reshape((28,28))
print one_hot(train_y[57],10)

print "---------valid"

print valid_x[57].reshape((28,28))
print one_hot(valid_y[57],10)

print "--------test"

print test_x[57].reshape((28,28))
print one_hot(test_y[57],10)

# TODO: the neural net!!

x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

W1 = tf.Variable(np.float32(np.random.rand(784, 10)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(10, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))
#loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20
train_size = int(len(train_x))  # Training size
validation_size = int(len(valid_x))  # Validation size = 15% out of total size
test_size = int(len(valid_x))  # test size = 15% out of total size
errorRef = 0;
contadorNoMejora = 0;
outfile = open('errores.txt', 'w')
outfile2 = open('errores2.txt', 'w')
epoch=0

#for epoch in xrange(1000):
while (contadorNoMejora < 6) or (epoch <= 20):
    for jj in xrange(int(train_size / batch_size)):
        minimo = min(batch_size, train_size - jj * batch_size)
        batch_xs = train_x[jj * batch_size: jj * batch_size + minimo]
        batch_ys = y_data_train[jj * batch_size: jj * batch_size + minimo]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    batch_xv = valid_x
    batch_yv = y_data_valid
    error = sess.run(loss, feed_dict={x: batch_xv, y_: batch_yv})
    outfile.write(str(error) + "\n");
    outfile2.write(str(cuenta_errores(batch_xv, batch_yv)) + "\n");

    if (epoch == 0):
        errorRef = error
    elif (error >= errorRef * 0.99):  # rango de mejora relativa debe ser mayor que 5%
        contadorNoMejora = contadorNoMejora + 1
    else:
        contadorNoMejora = 0
        errorRef = error

    print "Epoch #:", epoch, "Error: ", error, "         Contador no mejora: ", contadorNoMejora
    print "----------------------------------------------------------------------------------"
    epoch += 1
    #f (contadorNoMejora >= 6 and epoch >= 20):  # Minimo 20 iteraciones
    #    break

outfile.close()
outfile2.close()
batch_xt = test_x
batch_yt = y_data_test

print "Numero de errores. Conjunto Test:"
print cuenta_errores(batch_xt, batch_yt)
print "Porcentaje:"
print float(cuenta_errores(batch_xt, batch_yt))/test_size




























