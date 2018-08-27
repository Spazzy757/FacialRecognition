import tensorflow as tf
import pandas as pd
import numpy as np
import os.path
import time
import os


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


class BreedClassifier(object):

    def __init__(self):
        # TensorFlow configuration/initialization
        model_file = "media/models/breed_model.pb"
        self.label_file = "media/models/breed_labels.txt"
        self.input_height = 299
        self.input_width = 299
        self.input_mean = 128
        self.input_std = 128
        input_layer = "Mul"
        output_layer = "final_result"

        # Load TensorFlow Graph from disk
        self.graph = self.load_graph(model_file)

        # Grab the Input/Output operations
        input_name = "import/" + input_layer
        output_name = "import/" + output_layer

        self.input_operation = self.graph.get_operation_by_name(input_name)
        self.output_operation = self.graph.get_operation_by_name(output_name)

    def classify(self, file_name):
        t = self.read_tensor_from_image_file(
            file_name,
            input_height=self.input_height,
            input_width=self.input_width,
            input_mean=self.input_mean,
            input_std=self.input_std
        )

        with tf.Session(graph=self.graph) as sess:
            start = time.time()

            results = sess.run(
                self.output_operation.outputs[0],
                {
                    self.input_operation.outputs[0]: t
                }
            )

            end = time.time()

            results = np.squeeze(results)
            print(results.argsort())
            top_k = results.argsort()[-5:][::-1]

            labels = self.load_labels(self.label_file)

        print('\nEvaluation time (1-image): {:.3f}s\n'.format(end - start))

        for i in top_k:
            print(labels[i], results[i])
        data = {
            'labels': labels,
            'results': results.tolist(),
            'classification_time': end - start
        }
        return data

    @staticmethod
    def load_graph(model_file):
        graph = tf.Graph()
        graph_def = tf.GraphDef()

        with open(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)
        return graph

    @staticmethod
    def read_tensor_from_image_file(
            file_name,
            input_height=299,
            input_width=299,
            input_mean=0,
            input_std=255
    ):
        input_name = "file_reader"

        output_name = "normalized"

        file_reader = tf.read_file(file_name, input_name)

        if file_name.endswith(".png"):
            image_reader = tf.image.decode_png(
                file_reader,
                channels=3,
                name='png_reader'
            )
        elif file_name.endswith(".gif"):
            image_reader = tf.squeeze(
                tf.image.decode_gif(
                    file_reader,
                    name='gif_reader'
                )
            )
        elif file_name.endswith(".bmp"):
            image_reader = tf.image.decode_bmp(
                file_reader,
                name='bmp_reader'
            )
        else:
            image_reader = tf.image.decode_jpeg(
                file_reader,
                channels=3,
                name='jpeg_reader'
            )

        float_caster = tf.cast(image_reader, tf.float32)

        dims_expander = tf.expand_dims(float_caster, 0)

        resized = tf.image.resize_bilinear(
            dims_expander,
            [
                input_height,
                input_width
            ]
        )
        normalized = tf.divide(
            tf.subtract(
                resized,
                [
                    input_mean
                ]
            ),
            [
                input_std
            ]
        )

        # Run TF Session
        sess = tf.Session()
        result = sess.run(normalized)
        return result

    @staticmethod
    def load_labels(label_file):
        label = []
        proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()

        for l in proto_as_ascii_lines:
            label.append(l.rstrip())

        return label


def set_clean_dataset():
    dataset_labels = pd.read_csv('dataset/labels.csv')
    count = 0
    for i in range(len(dataset_labels)):
        print('moving {} into directory {}'.format(
            dataset_labels.values[i][0],
            dataset_labels.values[i][1],
        ))
        #  Make Directory if it does not exist
        if not os.path.exists('media/dataset/{}'.format(
                dataset_labels.values[i][1])):
            os.makedirs('media/dataset/{}'.format(
                dataset_labels.values[i][1]))
        # Move image into correct directory if it doesn't exist
        if os.path.exists('dataset/train/{}.jpg'.format(
                dataset_labels.values[i][0]
        )):
            os.rename(
                'dataset/train/{}.jpg'.format(
                    dataset_labels.values[i][0]
                ),
                'media/dataset/{}/{}.jpg'.format(
                    dataset_labels.values[i][1],
                    dataset_labels.values[i][0]
                )
            )
            count += 1


