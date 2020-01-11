import tensorflow as tf
import numpy as np
import time
import sys

num_classes = 100
batch_size = 107
num_example_per_epoch = 128721
num_batches_per_epoch = (num_example_per_epoch / batch_size)        #40,036

examples_per_shard = 1030
input_queue_memory_factor = 3
num_worker = 1
sub_batch_size = int(batch_size / num_worker)


tf_train_record_pattern = "/home/tidb/Desktop/tfrecord_subclasses_train/train-*"
tf_val_record_pattern = "/home/tidb/Desktop/tfrecord_subclasses_val/val-*"



def parse_example_proto(example_serialized):
    """Parses an Example proto containing a training example of an image.
    The output of the build_image_data.py image preprocessing script is a dataset
    containing serialized Example protocol buffers. Each Example proto contains
    the following fields:
    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>
    Args:
        example_serialized: scalar Tensor tf.string containing a serialized Example protocol buffer.
    Returns:
        image_buffer: Tensor tf.string containing the contents of a JPEG file.
        label: Tensor tf.int32 containing the label.
    """
    # Dense features in Example proto.
    feature_map = {
        'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/class/label': tf.io.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
    }
    features = tf.parse_single_example(example_serialized, feature_map)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)
    return features['image/encoded'], label


def decode_jpeg(image_buffer, scope=None):
    """Decode a JPEG string into one 3-D float image Tensor.
    Args:
        image_buffer: scalar string Tensor.
        scope: Optional scope for name_scope.
    Returns:
        3-D float Tensor with values ranging from [0, 1).
    """
    with tf.name_scope(values=[image_buffer], name=scope, default_name='decode_jpeg'):
        # Decode the string as an RGB JPEG.
        # Note that the resulting image contains an unknown height and width
        # that is set dynamically by decode_jpeg. In other words, the height
        # and width of image is unknown at compile-time.
        image = tf.image.decode_jpeg(image_buffer, channels=3)
        # After this point, all image pixels reside in [0,1)
        # until the very end, when they're rescaled to (-1, 1).  The various
        # adjust_* ops all require this range for dtype float.
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image


def batch_inputs(dataset, batch_size, train, num_preprocess_threads=None, num_readers=1):
    """Contruct batches of training or evaluation examples from the image dataset.
    Args:
        dataset: instance of Dataset class specifying the dataset.
          See dataset.py for details.
        batch_size: integer
        train: boolean
        num_preprocess_threads: integer, total number of preprocessing threads
        num_readers: integer, number of parallel readers
    Returns:
        images: 4-D float Tensor of a batch of images
        labels: 1-D integer Tensor of [batch_size].
    Raises:
        ValueError: if data is not found
    """
    with tf.name_scope('batch_processing'):
        # Create filename_queue
        if train:
            filename_queue = tf.train.string_input_producer(dataset, shuffle=False, capacity=16)
        else:
            filename_queue = tf.train.string_input_producer(dataset, shuffle=False, capacity=1)
        # Approximate number of examples per shard.
        # Size the random shuffle queue to balance between good global
        # mixing (more examples) and memory use (fewer examples).
        # 1 image uses 299*299*3*4 bytes = 1MB
        # The default input_queue_memory_factor is 16 implying a shuffling queue
        # size: examples_per_shard * 16 * 1MB = 17.6GB
        min_queue_examples = examples_per_shard * input_queue_memory_factor
        if train:
            #examples_queue = tf.RandomShuffleQueue(
            examples_queue = tf.queue.FIFOQueue(
              capacity=min_queue_examples + 3 * batch_size,
              #min_after_dequeue=min_queue_examples,
              dtypes=[tf.string])
        else:
            examples_queue = tf.queue.FIFOQueue(
              capacity=examples_per_shard + 3 * batch_size,
              dtypes=[tf.string])
        # Create multiple readers to populate the queue of examples.
        if num_readers > 1:
            enqueue_ops = []
            for _ in range(num_readers):
                reader = tf.TFRecordReader()
                _, value = reader.read(filename_queue)
                enqueue_ops.append(examples_queue.enqueue([value]))
            tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
            example_serialized = examples_queue.dequeue()
        else:
            reader = tf.TFRecordReader()
            _, example_serialized = reader.read(filename_queue)
        # Reshape images into these desired dimensions.
        height = 256
        width = 256
        depth = 3
        images_and_labels = []
        for thread_id in range(num_preprocess_threads):
            # Parse a serialized Example proto to extract the image and metadata.
            image_buffer, label_index = parse_example_proto(example_serialized)
            image = decode_jpeg(image_buffer)
            image.set_shape([height, width, 3])
            images_and_labels.append([image, label_index])
        images, label_index_batch = tf.train.batch_join(
        #images, label_index_batch=tf.train.batch(
            images_and_labels,
            batch_size=batch_size,
            capacity=2 * num_preprocess_threads * batch_size)
        images = tf.cast(images, tf.float32)
        images = tf.reshape(images, shape=[batch_size, height, width, depth])
        return images, tf.reshape(label_index_batch, [batch_size])



train_dataset = tf.io.gfile.glob(tf_train_record_pattern)
train_images, train_labels = batch_inputs(train_dataset, batch_size, train=True, num_preprocess_threads=2, num_readers=2)
#train_labels_onehot = tf.one_hot(train_labels, num_classes, on_value=1, off_value=0, axis=1)

print(train_images.get_shape())
print(train_labels.get_shape())


val_dataset = tf.io.gfile.glob(tf_val_record_pattern)
val_images, val_labels = batch_inputs(val_dataset, sub_batch_size, train=False, num_preprocess_threads=1, num_readers=1)
#val_labels_onehot = tf.one_hot(val_labels, num_classes, on_value=1, off_value=0, axis=1)

print(val_images.get_shape())
print(val_labels.get_shape())


#sess = tf.compat.v1.train.MonitoredTrainingSession(master=server.target, is_chief=(task_number == 0), hooks=[sync_replicas_hook])
sess=tf.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess)
t_images, t_labels = sess.run([train_images, train_labels])

print(t_images)
print(t_labels)