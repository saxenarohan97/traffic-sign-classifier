prob_value = 0.75

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
X_train, y_train           = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test             = mnist.test.images, mnist.test.labels

assert(len(X_train) == len(y_train))
assert(len(X_validation) == len(y_validation))
assert(len(X_test) == len(y_test))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))
print("Test Set:       {} samples".format(len(X_test)))

print("Data successfully loaded")

import numpy as np

# Pad images with 0s
X_train      = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_validation = np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')

print("Updated Image Shape: {}".format(X_train[0].shape))

from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)

import tensorflow as tf

EPOCHS = 100
BATCH_SIZE = 512

##################################################################################################################################

from tensorflow.contrib.layers import flatten

def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 1

    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 32x32x3.
    weight0 = tf.Variable(tf.truncated_normal([1, 1, 1, 3], mu, sigma))
    bias0 = tf.Variable(tf.zeros([3]))

    conv0 = tf.nn.conv2d(x, weight0, [1, 1, 1, 1], padding = 'VALID')
    conv0 = tf.nn.bias_add(conv0, bias0)

    # TODO: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    weight1 = tf.Variable(tf.truncated_normal([5, 5, 3, 6], mu, sigma))
    bias1 = tf.Variable(tf.zeros([6]))

    conv1 = tf.nn.conv2d(conv0, weight1, [1, 1, 1, 1], padding = 'VALID')
    conv1 = tf.nn.bias_add(conv1, bias1)

    conv1 = tf.nn.tanh(conv1)
    conv1 = tf.nn.dropout(conv1, keep_prob=prob)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    weight2 = tf.Variable(tf.truncated_normal([5, 5, 6, 16], mu, sigma))
    bias2 = tf.Variable(tf.zeros([16]))

    conv2 = tf.nn.conv2d(pool1, weight2, [1, 1, 1, 1], padding='VALID')
    conv2 = tf.nn.bias_add(conv2, bias2)

    # TODO: Activation.
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.dropout(conv2, keep_prob = prob)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1,2,2,1], 'VALID')

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    flat = tf.contrib.layers.flatten(pool2)

    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    weight3 = tf.Variable(tf.truncated_normal([400, 120], mu, sigma))
    bias3 = tf.Variable(tf.zeros([120]))

    full1 = tf.add(tf.matmul(flat, weight3), bias3)

    # TODO: Activation.
    full1 = tf.nn.tanh(full1)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    weight4 = tf.Variable(tf.truncated_normal([120, 84], mu, sigma))
    bias4 = tf.Variable(tf.zeros([84]))

    full2 = tf.add(tf.matmul(full1, weight4), bias4)

    # TODO: Activation.
    full2 = tf.nn.sigmoid(full2)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    weight5 = tf.Variable(tf.truncated_normal([84, 10], mu, sigma))
    bias5 = tf.Variable(tf.zeros([10]))

    logits = tf.add(tf.matmul(full2, weight5), bias5)

    return logits

##################################################################################################################################

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)

prob = tf.placeholder(tf.float32, (None))

rate = tf.placeholder(tf.float32, (None))
alpha = 0.00001
initial_alpha = 0.00001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, prob: 1., rate: alpha})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

from tqdm import tqdm

print('1. Start training afresh')
print('2. Continue from last saved state')
init_method = input('Choice: ')

with tf.Session() as sess:

    if init_method == '1':
        sess.run(tf.global_variables_initializer())

    elif init_method == '2':
        saver.restore(sess, './lenet')

    num_examples = len(X_train)

    tqdm.write("Training...")
    tqdm.write('')

    total = EPOCHS * (num_examples//BATCH_SIZE + 1)
    pbar = tqdm(total = EPOCHS * (num_examples//BATCH_SIZE))

    count = 0

    try:

        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, prob: prob_value, rate: alpha})
		
                count += 1
                alpha = initial_alpha * (total - count)/total
                pbar.update(1)

            validation_accuracy = evaluate(X_validation, y_validation)
            tqdm.write('Epoch {}: {}'.format(i+1, validation_accuracy))

    finally:
        choice = input('Save the model (y/n)? ')

        if choice == 'y':
            saver.save(sess, './lenet')
            print("Model saved")

