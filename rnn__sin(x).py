import numpy as np
import tensorflow as tf

import matplotlib as mpl
mpl.use('Qt5Agg')
from matplotlib import pyplot as plt

HIDDEN_SIZE = 30        #LSTM隐藏节点个数
NUM_LAYERS = 2      #LSTM的层数

TIMESTEPS = 10      #rnn训练序列长度
TRAINING_STEPS = 10000      #训练轮数
BATCH_SIZE = 32     #batch大小

TRAINING_EXAMPLES = 10000       #训练数据个数
TESTING_EXAMPLES = 1000         #测试数据个数
SAMPLE_GAP = 0.01       #采样间隔


def generate_data(seq):
    X = []
    y = []

    for i in range(len(seq) - TIMESTEPS):
        X.append([seq[i: i + TIMESTEPS]])
        y.append([seq[i + TIMESTEPS]])
    return np.array(X,dtype=np.float32),np.array(y,dtype=np.float32)

def lstm_model(X,y,is_training):
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])

    outputs, _ = tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)

    output = outputs[:,-1,:]

    predictions = tf.contrib.layers.fully_connected(output,1,activation_fn=None)

    if not is_training:
        return predictions,None,None

    loss = tf.losses.mean_squared_error(labels=y,predictions=predictions)

    train_op = tf.contrib.layers.optimize_loss(loss,tf.train.get_global_step(),\
        optimizer="Adagrad",learning_rate=0.1)
    return predictions,loss,train_op

def train(sess,train_X,train_y):
    ds = tf.data.Dataset.from_tensor_slices((train_X,train_y))
    ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE)
    iterator = ds.make_initializable_iterator()
    X,y = iterator.get_next()
#    X,y = ds.make_initializable_iterator().get_next()

    with tf.variable_scope("model"):
        predictions,loss,train_op = lstm_model(X,y,True)

    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer)          #初始化iterator
    for i in range(TRAINING_STEPS):
        _, l = sess.run([train_op,loss])
        if i % 100 == 0:
            print("training step: " + str(i) + ",loss: " + str(l))

def run_eval(sess,test_X,test_y):
    ds = tf.data.Dataset.from_tensor_slices((test_X,test_y))
    ds = ds.batch(1)
    X, y = ds.make_one_shot_iterator().get_next()

    with tf.variable_scope("model",reuse = True):
        prediction,_,_ = lstm_model(X,[0.0],False)

    predictions = []
    labels = []
    for i in range(TESTING_EXAMPLES):
        p, l = sess.run([prediction,y])
        predictions.append(p)
        labels.append(l)

    predictions = np.array(predictions).squeeze()
    labels = np.array(labels).squeeze()
    rmse = np.sqrt(((predictions - labels) ** 2).mean(axis=0))
    print("Mean Square Error is : %f" % rmse)

    plt.figure()
    print("!!!!!")
    plt.plot(predictions,label='predictions')
    plt.plot(labels,label='real_sin')
#    plt.plot(predictions,labels)
#    plt.xlabel('predictions')
#    plt.ylabel('labels')
    plt.legend()
    plt.show()


test_start = (TRAINING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
test_end = test_start + (TESTING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
train_X,train_y = generate_data(np.sin(np.linspace(0,test_start,\
    TRAINING_EXAMPLES+TIMESTEPS,dtype=np.float32)))
test_X,test_y = generate_data(np.sin(np.linspace(test_start,\
    test_end,TESTING_EXAMPLES+TIMESTEPS,dtype=np.float32)))



if __name__ == '__main__':
    print("test_start: ",test_start)
    print("test_end: ",test_end)
    print("train_X: ",train_X)
    print("train_y: ",train_y)
    print("test_X: ",test_X)
    print("test_y: ",test_y)


    with tf.Session() as sess:
        train(sess,train_X,train_y)
        run_eval(sess,test_X,test_y)
