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

data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data
x_data = data[:, 0:4].astype('f4')  # the samples are the four first rows of data
y_data = one_hot(data[:, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code

print "\nSome samples..."
for i in range(20):
    print x_data[i], " -> ", y_data[i]
print

x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20
train_size= int(len(x_data)*0.7) # Training size = 70% out of total size
validation_size= int(len(x_data)*0.15) # Validation size = 15% out of total size
test_size= len(x_data) - train_size - validation_size # test size = 15% out of total size
errorRef = 0;
contadorNoMejora = 0;
outfile = open('errores.txt', 'w')
outfile2 = open('errores2.txt', 'w')
epoch=0

#for epoch in xrange(1000):
while (contadorNoMejora < 6) or (epoch <= 20):

    for jj in xrange(int(train_size / batch_size)):
        minimo = min(batch_size, train_size - jj * batch_size)
        batch_xs = x_data[jj * batch_size: jj * batch_size + minimo]
        batch_ys = y_data[jj * batch_size: jj * batch_size + minimo]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    batch_xv = x_data[jj*batch_size + minimo : jj*batch_size + minimo + validation_size]
    batch_yv = y_data[jj*batch_size + minimo : jj*batch_size + minimo + validation_size]
    error = sess.run(loss, feed_dict={x: batch_xv, y_: batch_yv})
    outfile.write(str(error) + "\n");
    outfile2.write(str(cuenta_errores(batch_xv,batch_yv)) + "\n");

    if (epoch == 0):
        errorRef = error
    elif (error >= errorRef * 0.95):  # rango de mejora relativa debe ser mayor que 5%
        contadorNoMejora = contadorNoMejora + 1
    else:
        contadorNoMejora = 0
        errorRef = error

    print "Epoch #:", epoch, "Error: ", error, "          Contador no mejora: ",contadorNoMejora
    result = sess.run(y, feed_dict={x: batch_xv})
    for b, r in zip(batch_yv, result):
        print b, "-->", r
    print "----------------------------------------------------------------------------------"
    epoch += 1
    #if (contadorNoMejora >= 6 and epoch >= 20): # Minimo 20 iteraciones
	#    break

outfile.close()
outfile2.close()
batch_xt = x_data[len(x_data) - test_size:]
batch_yt = y_data[len(y_data) - test_size:]



print "Numero de errores. Conjunto Test:"
print cuenta_errores(batch_xt,batch_yt)
print "Porcentaje:"
print float(cuenta_errores(batch_xt, batch_yt))/test_size













