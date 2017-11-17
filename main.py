import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf 

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

# load data
trainX = mnist.train.images
trainYunparsed = mnist.train.labels
testX = mnist.test.images
testYunparsed = mnist.test.labels

# reformulate the labels
trainY = np.zeros((trainYunparsed.shape[0], 10))
testY = np.zeros((testYunparsed.shape[0], 10))
for i in range(0, trainYunparsed.shape[0]):
	trainY[i][trainYunparsed[i]] = 1
for i in range(0, testYunparsed.shape[0]):
	testY[i][testYunparsed[i]] = 1

# shuffling the training data
shuffleX = np.zeros_like(trainX)
shuffleY = np.zeros_like(trainY)

ran = np.arange(0,trainX.shape[0])
np.random.shuffle(ran)
shuffleX[:] = trainX[ran[:]]
shuffleY[:] = trainY[ran[:]]

trainX = np.copy(shuffleX)
trainY = np.copy(shuffleY)

# Parameters
learning_rate = 0.1
training_epochs = 2000
batch_size = 128
display_step = 50

## Network Parameters
n_hidden_1 = 1000 	# 1st layer number of units
n_hidden_2 = 50	# 2nd layer number of units
num_input = 28*28 	# MNIST data input (img shape: 28*28)
num_classes = 10 	# MNIST total classes (0-9 digits)

X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])



# # Store layers weight & bias
weights = {
	'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
	'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
	}

biases = {
	'b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'b2': tf.Variable(tf.random_normal([n_hidden_2])),
	'out': tf.Variable(tf.random_normal([num_classes]))
}

# Create model
def neural_net(x, weights, biases):
	# Hidden fully connected layer with 256 neurons
	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	layer_1 = tf.nn.sigmoid(layer_1)
	# Hidden fully connected layer with 256 neurons
	layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
	layer_2 = tf.nn.sigmoid(layer_2)
	# Output fully connected layer with a neuron for each class
	out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
	# out_layer = tf.nn.sigmoid(out_layer)
	return out_layer

predictions = neural_net(X, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

cost_graph = []
# Training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # train training_epoch times
    for epoch in range(training_epochs):
        avg_cost = 0.0
        total_batch = int(len(trainX) / batch_size)

        # Splitting into batches
        x_batches = np.array_split(trainX, total_batch)
        y_batches = np.array_split(trainY, total_batch)

        # For each batch
        for i in range(total_batch):
            batch_x, batch_y = x_batches[i], y_batches[i]
            o, c = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y})
            avg_cost += c / total_batch
            cost_graph.append(avg_cost)
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
    print("Optimization Finished!")
    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: testX, Y: testY}))

    # predict = np.argmax(sess.run([predictions], feed_dict={X: testX})[0],1)
    # ground_truth = np.argmax(testY, 1)

    # for i in range(predict.shape[0]):
    # 	if i % 10 == 0:
    # 		print("Ground_truth\tprediction")
    # 	print(ground_truth[i],"\t\t",predict[i])


plt.plot(cost_graph, 'ro-')
plt.title('error over t')
plt.savefig("./Results/mnist.png")
plt.show()