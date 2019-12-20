import tensorflow as tf
import  os
import numpy as np
import pickle
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import batch_norm
import tensorflow.contrib as tc
import warnings
warnings.simplefilter('ignore')

CIFAR_DIR = "./cifar-10-batches-py"

def load_data( filename ):
    '''read data from data file'''
    with open( filename, 'rb' ) as f:
        data = pickle.load( f, encoding='bytes' )
        return data[b'data'], data[b'labels']

class CifarData:
    def __init__( self, filenames, need_shuffle ):
        all_data = []
        all_labels = []
        for filename in filenames:
            data, labels = load_data( filename )
            all_data.append( data )
            all_labels.append( labels )
        self._data = np.vstack(all_data)
        self._data = self._data / 127.5 - 1
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

train_filename = [os.path.join(CIFAR_DIR, 'data_batch_%d' % i) for i in range(1, 6)]
test_filename = [os.path.join(CIFAR_DIR, 'test_batch')]
train_data = CifarData( train_filename, True )
test_data = CifarData( test_filename, False )


def separable_conv_block(x,  output_channel_number, name, is_train):
    '''
    mobilenet 卷积块
    :param x:
    :param output_channel_number:  输出通道数量 output channel of 1*1 conv layer
    :param name:
    :is_train: 是否进行卷积
    :return:
    '''
    with tf.variable_scope(name):
        input_channel = x.get_shape().as_list()[-1]
        # channel_wise_x: [channel1, channel2, ...]
        channel_wise_x = tf.split(x, input_channel, axis = 3)
        output_channels = []
        for i in range(len(channel_wise_x)):
            output_channel = tf.layers.conv2d(channel_wise_x[i],
                                              1,
                                              (3, 3),
                                              strides = (1, 1),
                                              padding = 'same',
                                              activation = None,
                                              name = 'conv_%d' % i
                                              )
            bn = tf.layers.batch_normalization(output_channel, training = is_train)
            new_output_channel = tf.nn.relu(bn)
            output_channels.append(new_output_channel)
        concat_layer = tf.concat(output_channels, axis = 3)
        conv1_1 = tf.layers.conv2d(concat_layer,
                                   output_channel_number,
                                   (1, 1),
                                   strides = (1, 1),
                                   padding = 'same',
                                   activation = None,
                                   name = name+'/conv1_1'
                                   )
        bn = tf.layers.batch_normalization(conv1_1, training = is_train)
        return tf.nn.relu(bn)


def inverted_bottleneck(idx, input, up_sample_rate, channels, subsample, is_train):
    normalizer = tc.layers.batch_norm
    bn_params = {'is_training': is_train}
    with tf.variable_scope('inverted_bottleneck{}_{}_{}'.format(idx, up_sample_rate, subsample)):
        stride = 2 if subsample else 1
        output = tc.layers.conv2d(input, up_sample_rate*input.get_shape().as_list()[-1], 1,
                                  activation_fn=tf.nn.relu6,
                                  normalizer_fn=normalizer, normalizer_params=bn_params)
        output = tc.layers.separable_conv2d(output, None, 3, 1, stride=stride,
                                            activation_fn=tf.nn.relu6,
                                            normalizer_fn=normalizer, normalizer_params=bn_params)
        output = tc.layers.conv2d(output, channels, 1, activation_fn=None,
                                  normalizer_fn=normalizer, normalizer_params=bn_params)
        if input.get_shape().as_list()[-1] == channels:
            output = tf.add(input, output)
        return output


IMPCLAS=10
x = tf.placeholder( tf.float32, [None, 3072] )
y = tf.placeholder( tf.int64, [None] )
is_train = tf.placeholder(tf.bool, [])

x_image = tf.reshape( x, [-1, 3, 32, 32] )
x_image = tf.transpose( x_image, perm= [0, 2, 3, 1] )   #x_image.shape == -1,32,32,3

# conv1 = tf.layers.conv2d(x_image, 32, ( 3, 3 ), padding = 'same', activation = tf.nn.relu, name = 'conv1')
# pooling1 = tf.layers.max_pooling2d(conv1, ( 2, 2 ), ( 2, 2 ), name='pool1')
# separable_2a = separable_conv_block(pooling1, 32, name = 'separable_2a', is_train = is_train)
# separable_2b = separable_conv_block(separable_2a, 32, name = 'separable_2b', is_train = is_train)
# pooling2 = tf.layers.max_pooling2d(separable_2b, ( 2, 2 ), ( 2, 2 ), name='pool2')
# separable_3a = separable_conv_block(pooling2, 32, name = 'separable_3a', is_train = is_train)
# separable_3b = separable_conv_block(separable_3a, 32, name = 'separable_3b', is_train = is_train)
# pooling3 = tf.layers.max_pooling2d(separable_3b, ( 2, 2 ), ( 2, 2 ), name='pool3')
# flatten = tf.contrib.layers.flatten(pooling3)
# y_ = tf.layers.dense(flatten, 10)
# print(y_.get_shape())

#output = tc.layers.conv2d(x, 32, 3, 2, normalizer_fn=tc.layers.batch_norm, normalizer_params={'is_training': is_train})
output = tf.layers.conv2d(x_image, 32, ( 3, 3 ), padding = 'same', activation = tf.nn.relu, name = 'conv1')
output = inverted_bottleneck(1,output, 1, 16, 0, is_train)
output = inverted_bottleneck(2,output, 6, 24, 1, is_train)
output = inverted_bottleneck(3,output, 6, 24, 0, is_train)
output = inverted_bottleneck(4,output, 6, 32, 1, is_train)
output = inverted_bottleneck(5,output, 6, 32, 0, is_train)
output = inverted_bottleneck(6,output, 6, 32, 0, is_train)
output = inverted_bottleneck(7,output, 6, 64, 1, is_train)
output = inverted_bottleneck(8,output, 6, 64, 0, is_train)
output = inverted_bottleneck(9,output, 6, 64, 0, is_train)
output = inverted_bottleneck(10,output, 6, 64, 0, is_train)
output = inverted_bottleneck(11,output, 6, 96, 0, is_train)
output = inverted_bottleneck(12,output, 6, 96, 0, is_train)
output = inverted_bottleneck(13,output, 6, 96, 0, is_train)
output = inverted_bottleneck(14,output, 6, 160, 1, is_train)
output = inverted_bottleneck(15,output, 6, 160, 0, is_train)
output = inverted_bottleneck(16,output, 6, 160, 0, is_train)
output = inverted_bottleneck(17,output, 6, 320, 0, is_train)
output = tc.layers.conv2d(output, 1280, 1, normalizer_fn=tc.layers.batch_norm, normalizer_params={'is_training': is_train})
flatten = tf.contrib.layers.flatten(output)
y_ = tf.layers.dense(flatten, 10)
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

batch_size = 20
train_steps = 100000
test_steps = 100
with tf.Session() as sess:
    sess.run( init )
    for i in range( train_steps ):
        batch_data, batch_labels = train_data.next_batch( batch_size )
        #print("batch shape: ",batch_data.shape, batch_labels.shape)
        loss_val, acc_val, _ = sess.run( [loss, accuracy, train_op], feed_dict={x:batch_data, y:batch_labels, is_train:True} )
        if ( i+1 ) % 200 == 0:
            print('[Train] Step: %d, loss: %4.5f, acc: %4.5f' % ( i+1, loss_val, acc_val ))
        if ( i+1 ) % 1000 == 0:
            test_data = CifarData( test_filename, False )
            all_test_acc_val = []
            for j in range( test_steps ):
                test_batch_data, test_batch_labels = test_data.next_batch( batch_size )
                test_acc_val = sess.run( [accuracy], feed_dict={ x:test_batch_data, y:test_batch_labels, is_train:False } )
                all_test_acc_val.append( test_acc_val )
            test_acc = np.mean( all_test_acc_val )
            print('[Test ] Step: %d, acc: %4.5f' % ( (i+1), test_acc ))


