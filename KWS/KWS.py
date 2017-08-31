from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import util.read_data as r
import datetime

# initial attributes
numcep = 26
mfcc_len = 199
batch_size = 10
label = 2
lr = 1e-5
epoch = 20
csv_file = 'record.csv'
data_path = './data/'
rd = r.Datas(numcep, batch_size, epoch, data_path, csv_file)


def stopwatch(start_duration=0):
    """This function will toggle a stopwatch.
    The first call starts it, second call stops it, third call continues it etc.
    So if you want to measure the accumulated time spent in a certain area of the code,
    you can surround that code by stopwatch-calls like this:

    .. code:: python

        fun_time = 0 # initializes a stopwatch
        [...]
        for i in range(10):
          [...]
          # Starts/continues the stopwatch - fun_time is now a point in time (again)
          fun_time = stopwatch(fun_time)
          fun()
          # Pauses the stopwatch - fun_time is now a duration
          fun_time = stopwatch(fun_time)
        [...]
        # The following line only makes sense after an even call of :code:`fun_time = stopwatch(fun_time)`.
        print 'Time spent in fun():', format_duration(fun_time)
    """
    if start_duration == 0:
        return datetime.datetime.utcnow()
    else:
        return datetime.datetime.utcnow() - start_duration


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def model():
    x = tf.placeholder(tf.float32, shape=[None, mfcc_len * numcep], name="input_source")  # mfcc feature(26,)
    x_image = tf.reshape(x, [-1, mfcc_len, numcep, 1])
    y_ = tf.placeholder(tf.float32, shape=[None, label], name="label")

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([50 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 50 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32,name="keep_prob")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, label])
    b_fc2 = bias_variable([label])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return x, y_, y_conv, keep_prob


def loss_and_accuracy(y_, y_conv):
    # training loss
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_soft), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  # prediction answer
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # accuacy calculate
    return cross_entropy, train_step, accuracy


def train_data(x, y_, keep_prob, train_step):
    for train_num in range(batch_num):  # training batch_size
        batch_x, label_train = rd.set_datas(train_num)
        train_step.run(feed_dict={x: batch_x, y_: label_train, keep_prob: 0.5})
        del batch_x[:], label_train[:]


def val_data(x, y_, keep_prob,cross_entropy, accuracy):
    total_loss = 0
    total_acc = 0
    for train_num in range(batch_num):  # training batch_size
        batch_x, label_train = rd.set_datas(train_num)
        loss = cross_entropy.eval(feed_dict={x: batch_x, y_: label_train, keep_prob: 1.0})
        total_loss += loss
        train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_: label_train, keep_prob: 1.0})
        total_acc += train_accuracy
        del batch_x[:], label_train[:]
    return total_loss, total_acc


def export(step):
    """Restores the trained variables into a simpler graph that will be exported for serving.
    """
    with tf.device('/cpu:0'):
        x, _, y_conv, keep_prob =  model()
        saver = tf.train.Saver()
        tf.global_variables_initializer().run()
        if step != None:
            save_path = saver.save(sess, "./model/model.ckpt", global_step=step)
        else:
            save_path = saver.save(sess, "./export/model.ckpt")

        print("Model saved in file: ", save_path)


sess = tf.InteractiveSession()
# visualize tensorflow
# writer = tf.summary.FileWriter("TensorBoard/", graph = sess.graph)
# initial variables
x, y_, y_conv, keep_prob = model()
cross_entropy, train_step, accuracy = loss_and_accuracy(y_, y_conv)
tf.global_variables_initializer().run()
batch_acc_x, _ = rd.set_acc_data()  # accuracy input all datas
batch_num = len(batch_acc_x) // batch_size  # input training batch

saver = tf.train.Saver()  # initialize saver for training model
training_time = stopwatch()

for step in range(epoch):
    train_data(x, y_, keep_prob, train_step)
    if step % 2 == 0 :
        total_loss, total_acc = val_data(x, y_, keep_prob, cross_entropy, accuracy)
        print("step: %d, loss: %g" % (step, total_loss/len(batch_acc_x)))
        print("step: %d, training accuracy: %g" % (step, total_acc/len(batch_acc_x)))
        if (step + 1) % epoch == 0:
            export_flag = None
        else:
            export_flag = step
        export(export_flag)
training_time = stopwatch(training_time)
print("end_training: {}".format(training_time))
