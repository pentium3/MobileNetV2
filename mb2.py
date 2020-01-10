import tensorflow as tf
import  os
import numpy as np
import pickle
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import batch_norm
import tensorflow.contrib as tc
import warnings
warnings.simplefilter('ignore')

from ops import *

# CIFAR_DIR = "./cifar-10-batches-py"

# def load_data( filename ):
#     '''read data from data file'''
#     with open( filename, 'rb' ) as f:
#         data = pickle.load( f, encoding='bytes' )
#         return data[b'data'], data[b'labels']

# class CifarData:
#     def __init__( self, filenames, need_shuffle ):
#         all_data = []
#         all_labels = []
#         for filename in filenames:
#             data, labels = load_data( filename )
#             all_data.append( data )
#             all_labels.append( labels )
#         self._data = np.vstack(all_data)
#         self._data = self._data / 127.5 - 1
#         self._labels = np.hstack( all_labels )
#         self._num_examples = self._data.shape[0]
#         self._need_shuffle = need_shuffle
#         self._indicator = 0
#         if self._need_shuffle:
#             self._shffle_data()
#     def _shffle_data( self ):
#         p = np.random.permutation( self._num_examples )
#         self._data = self._data[p]
#         self._labels = self._labels[p]
#     def next_batch( self, batch_size ):
#         '''return batch_size example as a batch'''
#         end_indictor = self._indicator + batch_size
#         if end_indictor > self._num_examples:
#             if self._need_shuffle:
#                 self._shffle_data()
#                 self._indicator = 0
#                 end_indictor = batch_size
#             else:
#                 raise Exception( "have no more examples" )
#         if end_indictor > self._num_examples:
#             raise Exception( "batch size is larger than all example" )
#         batch_data = self._data[self._indicator:end_indictor]
#         batch_labels = self._labels[self._indicator:end_indictor]
#         self._indicator = end_indictor
#         return batch_data, batch_labels


class DataSets:
    def data_preprocessing(self, x, value_dtype):
        x = x.astype(value_dtype)
        return (x / 127.5) - 1
    def __init__( self, filenames, need_shuffle ):
        all_data = []
        all_labels = []
        from tensorflow.keras.datasets import cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = self.data_preprocessing(x_train, "float64")
        #x_train = x_train.reshape((x_train.shape[0], 32*32*3))
        y_train = y_train.reshape((y_train.shape[0])).astype("int64")
        x_test = self.data_preprocessing(x_test, "float64")
        #x_test = x_test.reshape((x_test.shape[0], 32*32*3))
        y_test = y_test.reshape((y_test.shape[0])).astype("int64")
        if(filenames=='train'):
            all_data.append(x_train)
            all_labels.append(y_train)
        else:
            all_data.append(x_test)
            all_labels.append(y_test)
        self._data = np.vstack(all_data)
        #self._data = self._data / 127.5 - 1
        self._labels = np.hstack( all_labels )
        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._indicator = 0
        if self._need_shuffle:
            self._shffle_data()
    def _shffle_data( self ):
        p = np.random.permutation( self._num_examples )
        self._data = self._data[p]
        self._labels = self._labels[p]
    def next_batch( self, batch_size ):
        '''return batch_size example as a batch'''
        end_indictor = self._indicator + batch_size
        if end_indictor > self._num_examples:
            if self._need_shuffle:
                self._shffle_data()
                self._indicator = 0
                end_indictor = batch_size
            else:
                raise Exception( "have no more examples" )
        if end_indictor > self._num_examples:
            raise Exception( "batch size is larger than all example" )
        batch_data = self._data[self._indicator:end_indictor]
        batch_labels = self._labels[self._indicator:end_indictor]
        self._indicator = end_indictor
        return batch_data, batch_labels



#train_filename = [os.path.join(CIFAR_DIR, 'data_batch_%d' % i) for i in range(1, 6)]
#test_filename = [os.path.join(CIFAR_DIR, 'test_batch')]
# train_data_old = CifarData( train_filename, True )
# test_data_old = CifarData( test_filename, False )
train_data = DataSets( 'train', True )

batch_size = 32
train_steps = 60000
test_steps = 100
IMPCLAS=10
x = tf.placeholder( tf.float32, [batch_size, 32,32,3] )
y = tf.placeholder( tf.int64, [batch_size,] )
is_train = tf.placeholder(tf.bool, [])

# x_image = tf.reshape( x, [-1, 3, 32, 32] )
# x_image = tf.transpose( x_image, perm= [0, 2, 3, 1] )   #x_image.shape == -1,32,32,3

net = conv2d_block(x, 32, 3, 2, is_train, name='conv1_1')  # size/2

net = res_block(net, 1, 16, 1, is_train, name='res2_1')

net = res_block(net, 6, 24, 2, is_train, name='res3_1')  # size/4
net = res_block(net, 6, 24, 1, is_train, name='res3_2')

net = res_block(net, 6, 32, 2, is_train, name='res4_1')  # size/8
net = res_block(net, 6, 32, 1, is_train, name='res4_2')
net = res_block(net, 6, 32, 1, is_train, name='res4_3')

net = res_block(net, 6, 64, 1, is_train, name='res5_1')
net = res_block(net, 6, 64, 1, is_train, name='res5_2')
net = res_block(net, 6, 64, 1, is_train, name='res5_3')
net = res_block(net, 6, 64, 1, is_train, name='res5_4')

net = res_block(net, 6, 96, 2, is_train, name='res6_1')  # size/16
net = res_block(net, 6, 96, 1, is_train, name='res6_2')
net = res_block(net, 6, 96, 1, is_train, name='res6_3')

net = res_block(net, 6, 160, 2, is_train, name='res7_1')  # size/32
net = res_block(net, 6, 160, 1, is_train, name='res7_2')
net = res_block(net, 6, 160, 1, is_train, name='res7_3')

net = res_block(net, 6, 320, 1, is_train, name='res8_1', shortcut=False)

net = pwise_block(net, 1280, is_train, name='conv9_1')
net = global_avg(net)
y_ = flatten(conv_1x1(net, IMPCLAS, name='logits'))

print("output size:", y_.get_shape())


loss = tf.losses.sparse_softmax_cross_entropy( labels = y, logits = y_ )
predict = tf.argmax( y_, 1 )
correct_prediction = tf.equal( predict, y )
accuracy = tf.reduce_mean( tf.cast(correct_prediction, tf.float64) )

with tf.name_scope( 'train_op' ):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer( 1e-3 ).minimize( loss )

init = tf.global_variables_initializer()

with tf.Session(config=tf.ConfigProto(device_count={"CPU":12})) as sess:
    sess.run( init )
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range( train_steps ):
        batch_data, batch_labels = train_data.next_batch( batch_size )
        #print("batch shape: ",batch_data.shape, batch_labels.shape)
        loss_val, acc_val, _ = sess.run( [loss, accuracy, train_op], feed_dict={x:batch_data, y:batch_labels, is_train:True} )
        if ( i+1 ) % 200 == 0:
            print('[Train] Step: %d, loss: %4.5f, acc: %4.5f' % ( i+1, loss_val, acc_val ))
        if ( i+1 ) % 1000 == 0:
            #test_data = CifarData( test_filename, False )
            test_data = DataSets( 'test', False )
            all_test_acc_val = []
            for j in range( test_steps ):
                test_batch_data, test_batch_labels = test_data.next_batch( batch_size )
                test_acc_val = sess.run( [accuracy], feed_dict={ x:test_batch_data, y:test_batch_labels, is_train:False } )
                all_test_acc_val.append( test_acc_val )
            test_acc = np.mean( all_test_acc_val )
            print('[Test ] Step: %d, acc: %4.5f' % ( (i+1), test_acc ))
    
    coord.request_stop()
    coord.join(threads)
    sess.close()


# model_path="./pretrained"
# saver = tf.train.Saver()
# saver_path = saver.save(sess, model_path)
# print("Model saved in file:", saver_path)
