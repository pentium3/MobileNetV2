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

model_path = '/home/tidb/Desktop/MobileNetV2/mb_imgnet/pretrained'


from read_imgnet import tf_train_record_pattern, tf_val_record_pattern, batch_inputs


batch_size = 128
train_steps = 30000
test_steps = 100
PREIMPCLAS=100
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
lastlevel = global_avg(net)

pretrain_y_ = flatten(conv_1x1(lastlevel, PREIMPCLAS, name='logits_100'))

print("output size:", pretrain_y_.get_shape())


loss = tf.losses.sparse_softmax_cross_entropy( labels = y, logits = pretrain_y_ )
predict = tf.argmax( pretrain_y_, 1 )
correct_prediction = tf.equal( predict, y )
accuracy = tf.reduce_mean( tf.cast(correct_prediction, tf.float64) )

with tf.name_scope( 'train_op' ):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer( 1e-3 ).minimize( loss )

init = tf.global_variables_initializer()
train_dataset = tf.io.gfile.glob(tf_train_record_pattern)
t_images, t_labels = batch_inputs(train_dataset, batch_size, train=True, num_preprocess_threads=2, num_readers=2)
val_dataset = tf.io.gfile.glob(tf_val_record_pattern)
v_images, v_labels = batch_inputs(val_dataset, batch_size, train=False, num_preprocess_threads=1, num_readers=1)
print(t_images.get_shape())    # (107, 256, 256, 3)
print(t_labels.get_shape())    # (107,)


with tf.Session(config=tf.ConfigProto(device_count={"CPU":12})) as sess:
    sess.run( init )
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range( train_steps ):
        #print("batch")
        batch_data, batch_labels = sess.run([t_images, t_labels])
        #print(i, "batch")
        #print("batch shape: ",batch_data.shape, batch_labels.shape)
        loss_val, acc_val, _ = sess.run( [loss, accuracy, train_op], feed_dict={x:batch_data, y:batch_labels, is_train:True} )
        #print(i, "done")
        if ( i+1 ) % 200 == 0:
            print('[Train] Step: %d, loss: %4.5f, acc: %4.5f' % ( i+1, loss_val, acc_val ))
        if ( i+1 ) % 1000 == 0:
            all_test_acc_val = []
            for j in range( test_steps ):
                test_batch_data, test_batch_labels = sess.run([v_images, v_labels])
                test_acc_val = sess.run( [accuracy], feed_dict={ x:test_batch_data, y:test_batch_labels, is_train:False } )
                all_test_acc_val.append( test_acc_val )
            test_acc = np.mean( all_test_acc_val )
            print('[Test ] Step: %d, acc: %4.5f' % ( (i+1), test_acc ))
    
    coord.request_stop()
    coord.join(threads)

    saver = tf.compat.v1.train.Saver()
    saver_path = saver.save(sess, model_path)
    print("Model saved in file:", saver_path)


