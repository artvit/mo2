from functools import reduce

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import dataloader

dataset = dataloader.dataset_large
img_h = img_w = 28             # images are 28x28
img_size_flat = img_h * img_w  # 28x28=784, the total number of pixels
img_classes = 10

epochs = 20
batch_size = 128
display_freq = 256
learning_rate = 0.01
enable_dropout = True
dropout_rate = 0.15
regularization_enabled = False
regularization_beta = 0.01

number_of_units = 500
number_of_layers = 4

rnd_seed = 42


def map_result_to_arr(index):
    res = np.zeros((10,), dtype=np.float32)
    res[index] = 1.0
    return res


def load_data():
    data, results = dataloader.load_data(dataset)
    results = np.array([map_result_to_arr(r) for r in results], results.dtype)

    x_train, x_test, y_train, y_test = train_test_split(data, results, test_size=0.1, random_state=rnd_seed)
    x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=rnd_seed)

    return x_train, y_train, x_test, y_test, x_valid, y_valid


def weight_variable(name, shape):
    initial = tf.truncated_normal_initializer(stddev=0.01)

    return tf.get_variable('W_' + name,
                           dtype=tf.float32,
                           shape=shape,
                           initializer=initial)


def bias_variable(name, shape):
    initial = tf.constant(0., shape=shape, dtype=tf.float32)

    return tf.get_variable('b_' + name,
                           dtype=tf.float32,
                           initializer=initial)


def fc_layer(x, num_units, name, use_relu=True, dropout=enable_dropout):
    in_dim = x.get_shape()[1]

    w = weight_variable(name, shape=[in_dim, num_units])
    b = bias_variable(name, [num_units])

    layer = tf.matmul(x, w)
    layer += b

    if use_relu:
        layer = tf.nn.relu(layer)

    if dropout:
        layer = tf.nn.dropout(layer, keep_prob=1.0 - dropout_rate)

    return layer, w


def batch_shuffle(x, y, seed):
    if seed is not None:
        np.random.seed(seed)

    permutation = np.random.permutation(y.shape[0])
    x_shuffled = x[permutation, :]
    y_shuffled = y[permutation]

    return x_shuffled, y_shuffled


def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]

    return x_batch, y_batch


def init_nn(x, y):

    out_l = x
    weights = []
    for i in range(number_of_layers):
        layer, w = fc_layer(out_l, number_of_units, f'FC{i + 1}', True)
        out_l = layer
        weights.append(w)

    output_logits, w_out = fc_layer(out_l, img_classes, 'OUT', False, False)
    weights.append(w_out)

    class_prediction = tf.argmax(output_logits, axis=1, name='predictions')
    loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=output_logits),
                                   name='loss_function')
    if regularization_enabled:
        pass
        regularizer = reduce(lambda r1, r2: r1 + r2, map(tf.nn.l2_loss, weights))
        loss_function = tf.reduce_mean(loss_function + regularization_beta * regularizer)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name='Optimizer').minimize(loss_function)
    correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1), name='correct_prediction')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    init = tf.global_variables_initializer()

    return init, class_prediction, loss_function, optimizer, accuracy


def train(init, x, y, optimizer, loss_function, accuracy, tr_x, tr_y, v_x, v_y, rnd_seed):
    sess = tf.InteractiveSession()
    sess.run(init)
    global_step = 0
    num_tr_iter = int(len(tr_y) / batch_size)

    for epoch in range(epochs):
        tr_x, tr_y = batch_shuffle(tr_x, tr_y, rnd_seed)
        for iteration in tqdm(range(num_tr_iter), desc=f'Training epoch {epoch + 1}'):
            global_step += 1
            start = iteration * batch_size
            end = (iteration + 1) * batch_size
            x_batch, y_batch = get_next_batch(tr_x, tr_y, start, end)

            feed_dict_batch = {
                x: x_batch,
                y: y_batch
            }

            sess.run(optimizer, feed_dict=feed_dict_batch)

            # if iteration % display_freq == 0:
            #     loss_batch, acc_batch = sess.run([loss_function, accuracy], feed_dict=feed_dict_batch)
            #     print(f'iter {iteration:3d}:\t TRAIN Loss={loss_batch:.2f},\t Accuracy={acc_batch:.01%}')

        feed_dict_valid = {
            x: v_x[:1000],
            y: v_y[:1000]
        }

        loss_valid, acc_valid = sess.run([loss_function, accuracy], feed_dict=feed_dict_valid)

        print("Epoch: {0}, VALIDATION Loss: {1:.2f}, Accuracy: {2:.01%}".format(epoch + 1, loss_valid, acc_valid))
        print('-----------------------------------------------------')

    return sess


def test_nn(x, y, te_x, te_y, sess, loss_function, accuracy):
    feed_dict_valid = {
        x: te_x,
        y: te_y
    }

    loss_valid, acc_valid = sess.run([loss_function, accuracy], feed_dict=feed_dict_valid)

    print('---------------------------------------------------------')
    print("TEST Loss: {0:.2f}, Accuracy: {1:.01%}".format(loss_valid, acc_valid))
    print('---------------------------------------------------------')


def main():
    tr_x, tr_y, te_x, te_y, v_x, v_y = load_data()
    # tr_x = tr_x.reshape(tr_x.shape[0], -1)
    # te_x = te_x.reshape(te_x.shape[0], -1)
    # v_x = v_x.reshape(v_x.shape[0], -1)

    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='X')
    y = tf.placeholder(tf.float32, shape=[None, img_classes], name='Y')

    init, class_prediction, loss_function, optimizer, accuracy = init_nn(x, y)
    sess = train(init, x, y, optimizer, loss_function, accuracy, tr_x, tr_y, v_x, v_y, rnd_seed)
    test_nn(x, y, te_x, te_y, sess, loss_function, accuracy)


if __name__ == '__main__':
    main()
