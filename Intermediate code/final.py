import pickle
from pprint import pprint
import numpy as np
from tqdm import tqdm
from decimal import *

training_file = './traffic-signs-data/train.p'
testing_file = './traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

'''
images = X_train
labels = y_train

X_train = np.array([])
y_train = np.array([])

i = 0

while i != 38880:
    i = (int)(input())

    if i == 0:
        X_train = np.array([images[0]])
        y_train = np.array([labels[0]])

    else:
        X_train = np.append(X_train, [images[i]], 0)
        y_train = np.append(y_train, [labels[i]], 0)
'''

X_valid = X_test[:5052]
y_valid = y_test[:5052]

X_test = X_test[5052:]
y_test = y_test[5052:]

n_train = len(X_train)
n_test = len(X_test)
n_valid = len(X_valid)
image_shape = np.shape(X_train[0])
n_classes = 43

print("Number of training examples =", n_train)
print("Number of validation examples =", n_valid)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Preprocess the data here.
### Feel free to use as many code cells as needed.
X_train = X_train - np.mean(X_train)
X_train = X_train / np.std(X_train)

X_valid = X_valid - np.mean(X_valid)
X_valid = X_valid / np.std(X_valid)

X_test = X_test - np.mean(X_test)
X_test = X_test / np.std(X_test)

from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)

import tensorflow as tf

EPOCHS = 100
BATCH_SIZE = 1024

#####################################################################################################

from tensorflow.contrib.layers import flatten

def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # TODO: Layer 0: Convolutional. Input = 32x32x3. Output = 32x32x3.
    w0 = tf.Variable(tf.truncated_normal([1, 1, 3, 3], mean = mu, stddev = sigma))
    b0 = tf.Variable(tf.zeros([3]))
    conv0 = tf.nn.conv2d(x, w0, strides = [1, 1, 1, 1], padding = 'VALID')
    conv0 = tf.nn.bias_add(conv0, b0)

    # TODO: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x9.
    w1 = tf.Variable(tf.truncated_normal([5, 5, 3, 9], mean = mu, stddev = sigma))
    b1 = tf.Variable(tf.zeros([9]))
    conv1 = tf.nn.conv2d(conv0, w1, strides = [1, 1, 1, 1], padding = 'VALID')
    conv1 = tf.nn.bias_add(conv1, b1)

    # TODO: Activation.
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.dropout(conv1, keep_prob = prob)

    # TODO: Pooling. Input = 28x28x9. Output = 14x14x9.
    pool1 = tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')

    # TODO: Layer 2: Convolutional. Output = 10x10x27.
    w2 = tf.Variable(tf.truncated_normal([5, 5, 9, 27], mean = mu, stddev = sigma))
    b2 = tf.Variable(tf.zeros([27]))
    conv2 = tf.nn.conv2d(pool1, w2, strides = [1, 1, 1, 1], padding = 'VALID')
    conv2 = tf.nn.bias_add(conv2, b2)

    # TODO: Activation.
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.dropout(conv2, keep_prob = prob)

    # TODO: Pooling. Input = 10x10x27. Output = 5x5x27.
    pool2 = tf.nn.max_pool(conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')

    # TODO: Flatten. Input = 5x5x27. Output = 675.
    flat = flatten(pool2)

    # TODO: Layer 3: Fully Connected. Input = 675. Output = 120.
    w3 = tf.Variable(tf.truncated_normal([675, 120], mean = mu, stddev = sigma))
    b3 = tf.Variable(tf.zeros([120]))
    full1 = tf.matmul(flat, w3)
    full1 = tf.nn.bias_add(full1, b3)

    # TODO: Activation.
    full1 = tf.nn.relu(full1)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    w4 = tf.Variable(tf.truncated_normal([120, 84], mean = mu, stddev = sigma))
    b4 = tf.Variable(tf.zeros([84]))
    full2 = tf.matmul(full1, w4)
    full2 = tf.nn.bias_add(full2, b4)

    # TODO: Activation.
    full2 = tf.nn.relu(full2)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 43.
    w5 = tf.Variable(tf.truncated_normal([84, n_classes], mean = mu, stddev = sigma))
    b5 = tf.Variable(tf.zeros([n_classes]))
    logits = tf.matmul(full2, w5)
    logits = tf.nn.bias_add(logits, b5)

    return logits

####################################################################################################

### Train your model here.
### Feel free to use as many code cells as needed.

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
prob = tf.placeholder(tf.float32, (None))
one_hot_y = tf.one_hot(y, n_classes)

rate = tf.placeholder(tf.float32, (None))
alpha = 0.0001
initial_alpha = 0.0001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = alpha)
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
import sys

with tf.Session() as sess:

    choice = input('\n\nDo you want to: \n\n1. Start training afresh \n\
2. Continue from the last saved training \nEnter your response here: ')

    if choice == '1':

        sess.run(tf.global_variables_initializer())

    elif choice == '2':
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, './lenet')

    else:
        sys.exit()

    num_examples = len(X_train)

    tqdm.write("Training...")
    tqdm.write('\n')

    pbar = tqdm(total = EPOCHS * (len(X_train) + 1)//BATCH_SIZE)
    total = EPOCHS * (len(X_train) + 1)//BATCH_SIZE
    count = 0

    rate_list = np.array([])
    accuracy_x = np.array([])
    accuracy_y = np.array([])

    try:

        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)

            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, prob: 0.5, rate: alpha})

                count += 1
                alpha = initial_alpha * (total - count)/total
                rate_list = np.append(rate_list, alpha)

                pbar.update(1)

            validation_accuracy = evaluate(X_valid, y_valid)
            accuracy_x = np.append(accuracy_x, count)
            accuracy_y = np.append(accuracy_y, validation_accuracy)
            tqdm.write("{}".format(validation_accuracy))
	
    finally:
        choice = input('Save the model (y/n)? ')

        if choice == 'y':
            saver.save(sess, './lenet')
            np.save('./viz/rate', rate_list)
            np.save('./viz/accuracyx', accuracy_x)
            np.save('./viz/accuracyy', accuracy_y)
            tqdm.write("Model saved")

        else:
            tqdm.write('Session\'s training discarded')

        print("Learning rate: %f" % alpha)

