import os
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.app.flags.DEFINE_integer('training_iteration', 1000,
                            'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
FLAGS = tf.app.flags.FLAGS

def main(_):
    if len(sys.argv) < 2 or sys.argv[-1].startswith('-'):
        print('Usage: mnist_export.py [--training_iteration=x] '
              '[--model_version=y] export_dir')
        sys.exit(-1)
    if FLAGS.training_iteration <= 0:
        print('Please specify a positive value for training iteration.')
        sys.exit(-1)
    if FLAGS.model_version <= 0:
        print('Please specify a positive value for version number.')
        sys.exit(-1)

    sess = tf.InteractiveSession()
    images, scores = train(sess)
    export(sess, images, scores)

def train(sess):
    images = tf.placeholder('float', shape=[None, 28, 28, 1], name='images')
    x = tf.reshape(images, [-1, 784])
    y_ = tf.placeholder('float', shape=[None, 10])
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    sess.run(tf.global_variables_initializer())
    scores = tf.nn.softmax(tf.matmul(x, w) + b, name='scores')
    cross_entropy = -tf.reduce_sum(y_ * tf.log(scores))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    for _ in range(FLAGS.training_iteration):
        batch = mnist.train.next_batch(50)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    return (images, scores)

def export(sess, images, scores):
    export_path_base = sys.argv[-1]
    export_path = os.path.join(
        tf.compat.as_bytes(export_path_base),
        tf.compat.as_bytes(str(FLAGS.model_version)))

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    tensor_info_images = tf.saved_model.utils.build_tensor_info(images)
    tensor_info_scores = tf.saved_model.utils.build_tensor_info(scores)

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'images': tensor_info_images},
            outputs={'scores': tensor_info_scores},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict_images': prediction_signature
        },
        legacy_init_op=tf.group(tf.tables_initializer(), name='legacy_init_op'))
    builder.save(as_text=False)

if __name__ == '__main__':
  tf.app.run()
