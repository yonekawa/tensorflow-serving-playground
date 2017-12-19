import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.estimator.export import export
import sys
import os

tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
FLAGS = tf.app.flags.FLAGS

def main(_):
    if len(sys.argv) < 1 or sys.argv[-1].startswith('-'):
        print('Usage: keras_vgg.py export_dir')
        sys.exit(-1)

    export_path_base = sys.argv[-1]
    model = keras.applications.vgg16.VGG16(weights='imagenet')
    model.compile(optimizer=keras.optimizers.SGD(lr=.01, momentum=.9),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())
    estimator = tf.keras.estimator.model_to_estimator(keras_model=model)
    feature_spec = {'input_1': model.input}
    serving_input_fn = export.build_raw_serving_input_receiver_fn(feature_spec)
    estimator.export_savedmodel(export_path_base, serving_input_fn)

if __name__ == '__main__':
    tf.app.run()
