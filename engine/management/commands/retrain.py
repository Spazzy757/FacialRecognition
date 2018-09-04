from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from django.core.management.base import BaseCommand
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from datetime import datetime
import tensorflow as tf

from ..helpers import (
    get_random_distorted_bottlenecks,
    get_random_cached_bottlenecks,
    maybe_download_and_extract,
    add_final_training_ops,
    create_inception_graph,
    add_input_distortions,
    should_distort_images,
    add_evaluation_step,
    create_image_lists,
    cache_bottlenecks
)


# These are all parameters that are tied to the particular model architecture
# we're using for Inception v3. These include things like tensor names and their
# sizes. If you want to adapt this script to work with another model, you will
# need to update these to reflect the values in the network you're using.

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


class Command(BaseCommand):
    help = """Trains an Image Classification Model According to Inception
           v3 architecture model."""

    def add_arguments(self, parser):
        parser.add_argument(
            '--image_dir',
            type=str,
            default='',
            help='Path to folders of labeled images.'
        )
        parser.add_argument(
            '--output_graph',
            type=str,
            default='/tmp/output_graph.pb',
            help='Where to save the trained graph.'
        )
        parser.add_argument(
            '--output_labels',
            type=str,
            default='/tmp/output_labels.txt',
            help='Where to save the trained graph\'s labels.'
        )
        parser.add_argument(
            '--summaries_dir',
            type=str,
            default='/tmp/retrain_logs',
            help='Where to save summary logs for TensorBoard.'
        )
        parser.add_argument(
            '--how_many_training_steps',
            type=int,
            default=4000,
            help='How many training steps to run before ending.'
        )
        parser.add_argument(
            '--learning_rate',
            type=float,
            default=0.01,
            help='How large a learning rate to use when training.'
        )
        parser.add_argument(
            '--testing_percentage',
            type=int,
            default=10,
            help='What percentage of images to use as a test set.'
        )
        parser.add_argument(
            '--validation_percentage',
            type=int,
            default=10,
            help='What percentage of images to use as a validation set.'
        )
        parser.add_argument(
            '--eval_step_interval',
            type=int,
            default=10,
            help='How often to evaluate the training results.'
        )
        parser.add_argument(
            '--train_batch_size',
            type=int,
            default=100,
            help='How many images to train on at a time.'
        )
        parser.add_argument(
            '--test_batch_size',
            type=int,
            default=-1,
            help="""
          How many images to test on. This test set is only used once, to 
          evaluate the final accuracy of the model after training completes.
          A value of -1 causes the entire test set to be used, which leads to 
          more stable results across runs.
          """
        )
        parser.add_argument(
            '--validation_batch_size',
            type=int,
            default=100,
            help="""
          How many images to use in an evaluation batch. This validation set is
          used much more often than the test set, and is an early indicator of 
          how accurate the model is during training.
          A value of -1 causes the entire validation set to be used, which leads 
          to more stable results across training iterations, but may be slower 
          on large training sets.
          """
        )
        parser.add_argument(
            '--print_misclassified_test_images',
            default=False,
            help="""
          Whether to print out a list of all misclassified test images.
          """,
            action='store_true'
        )
        parser.add_argument(
            '--model_dir',
            type=str,
            default='/tmp/imagenet',
            help="""\
          Path to classify_image_graph_def.pb,
          imagenet_synset_to_human_label_map.txt, and
          imagenet_2012_challenge_label_map_proto.pbtxt.
          """
        )
        parser.add_argument(
            '--bottleneck_dir',
            type=str,
            default='/tmp/bottleneck',
            help='Path to cache bottleneck layer values as files.'
        )
        parser.add_argument(
            '--final_tensor_name',
            type=str,
            default='final_result',
            help="""\
          The name of the output classification layer in the retrained graph.
          """
        )
        parser.add_argument(
            '--flip_left_right',
            default=False,
            help="""
          Whether to randomly flip half of the training images horizontally.
          """,
            action='store_true'
        )
        parser.add_argument(
            '--random_crop',
            type=int,
            default=0,
            help="""
          A percentage determining how much of a margin to randomly crop off the
          training images.
          """
        )
        parser.add_argument(
            '--random_scale',
            type=int,
            default=0,
            help="""
          A percentage determining how much to randomly scale up the size of the
          training images by.
          """
        )
        parser.add_argument(
            '--random_brightness',
            type=int,
            default=0,
            help="""
          A percentage determining how much to randomly multiply the training 
          image input pixels up or down by.\
          """
        )

    def handle(self, *args, **options):
        # Setup the directory we'll write summaries to for TensorBoard
        if tf.gfile.Exists(options.get('summaries_dir')):
            tf.gfile.DeleteRecursively(options.get('summaries_dir'))

        tf.gfile.MakeDirs(options.get('summaries_dir'))

        # Set up the pre-trained graph.
        maybe_download_and_extract(options.get('model_dir'), data_url=DATA_URL)

        graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (
            create_inception_graph(options.get('model_dir'))
        )

        # Look at the folder structure, and create lists of all the images.
        image_lists = create_image_lists(
            options.get('image_dir'),
            options.get('testing_percentage'),
            options.get('validation_percentage')
        )

        class_count = len(image_lists.keys())

        if class_count == 0:
            print('No valid folders of images found at '
                  + options.get('image_dir'))
            return -1

        if class_count == 1:
            print('Only one valid folder of images found at '
                  + options.get('image_dir') +
                  ' - multiple classes are needed for classification.')
            return -1

        # See if the command-line flags mean we're applying any distortions.
        do_distort_images = should_distort_images(
            options.get('flip_left_right'),
            options.get('random_crop'),
            options.get('random_scale'),
            options.get('random_brightness')
        )

        with tf.Session(graph=graph) as sess:

            if do_distort_images:
                # We will be applying distortions, so setup the operations we'll
                # need.
                (
                    distorted_jpeg_data_tensor,
                    distorted_image_tensor

                ) = add_input_distortions(
                    options.get('flip_left_right'),
                    options.get('random_crop'),
                    options.get('random_scale'),
                    options.get('random_brightness')
                )
            else:
                # We'll make sure we've calculated the 'bottleneck' image
                # summaries and cached them on disk.
                cache_bottlenecks(
                    sess,
                    image_lists,
                    options.get('image_dir'),
                    options.get('bottleneck_dir'),
                    jpeg_data_tensor,
                    bottleneck_tensor
                )

            # Add the new layer that we'll be training.
            (
                train_step,
                cross_entropy,
                bottleneck_input,
                ground_truth_input,
                final_tensor
            ) = add_final_training_ops(
                options.get('learning_rate'),
                len(image_lists.keys()),
                options.get('final_tensor_name'),
                bottleneck_tensor
            )

            # Create the operations we need to evaluate the accuracy of our new
            # layer.
            evaluation_step, prediction = add_evaluation_step(
                final_tensor,
                ground_truth_input
            )

            # Merge all the summaries and write them out to the summaries_dir
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(
                options.get('summaries_dir') + '/train',
                sess.graph
            )

            validation_writer = tf.summary.FileWriter(
                options.get('summaries_dir') + '/validation'
            )

            # Set up all our weights to their initial default values.
            init = tf.global_variables_initializer()
            sess.run(init)

            # Run the training for as many cycles as requested on the command
            # line.
            for i in range(options.get('how_many_training_steps')):
                # Get a batch of input bottleneck values, either calculated
                # fresh every time with distortions applied, or from the cache
                # stored on disk.
                if do_distort_images:
                    (
                        train_bottlenecks,
                        train_ground_truth

                    ) = get_random_distorted_bottlenecks(
                        sess,
                        image_lists,
                        options.get('train_batch_size'),
                        'training',
                        options.get('image_dir'),
                        distorted_jpeg_data_tensor,
                        distorted_image_tensor,
                        resized_image_tensor,
                        bottleneck_tensor
                    )
                else:
                    (
                        train_bottlenecks,
                        train_ground_truth,
                        _
                    ) = get_random_cached_bottlenecks(
                        sess,
                        image_lists,
                        options.get('train_batch_size'),
                        'training',
                        options.get('bottleneck_dir'),
                        options.get('image_dir'),
                        jpeg_data_tensor,
                        bottleneck_tensor
                    )
                # Feed the bottlenecks and ground truth into the graph, and run
                # a training step. Capture training summaries for TensorBoard
                # with the `merged` op.

                train_summary, _ = sess.run(
                    [
                        merged,
                        train_step
                    ],
                    feed_dict={
                        bottleneck_input: train_bottlenecks,
                        ground_truth_input: train_ground_truth
                    }
                )
                train_writer.add_summary(train_summary, i)

                # Every so often, print out how well the graph is training.
                is_last_step = i + 1 == options.get('how_many_training_steps')

                if (i % options.get('eval_step_interval')) == 0 or is_last_step:
                    train_accuracy, cross_entropy_value = sess.run(
                        [
                            evaluation_step,
                            cross_entropy
                        ],
                        feed_dict={
                            bottleneck_input: train_bottlenecks,
                            ground_truth_input: train_ground_truth
                        }
                    )
                    print('%s: Step %d: Train accuracy = %.1f%%' % (
                        datetime.now(),
                        i,
                        train_accuracy * 100
                    ))
                    print('%s: Step %d: Cross entropy = %f' % (
                        datetime.now(),
                        i,
                        cross_entropy_value
                    ))
                    validation_bottlenecks, validation_ground_truth, _ = (
                        get_random_cached_bottlenecks(
                            sess,
                            image_lists,
                            options.get('validation_batch_size'),
                            'validation',
                            options.get('bottleneck_dir'),
                            options.get('image_dir'),
                            jpeg_data_tensor,
                            bottleneck_tensor
                        ))
                    # Run a validation step and capture training summaries for
                    # TensorBoard with the `merged` op.
                    validation_summary, validation_accuracy = sess.run(
                        [merged, evaluation_step],
                        feed_dict={
                            bottleneck_input: validation_bottlenecks,
                            ground_truth_input: validation_ground_truth
                        }
                    )
                    validation_writer.add_summary(validation_summary, i)

                    print('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' % (
                        datetime.now(),
                        i,
                        validation_accuracy * 100,
                        len(validation_bottlenecks)
                    ))

            # We've completed all our training, so run a final test evaluation
            # on some new images we haven't used before.
            test_bottlenecks, test_ground_truth, test_filenames = (
                get_random_cached_bottlenecks(
                    sess,
                    image_lists,
                    options.get('test_batch_size'),
                    'testing',
                    options.get('bottleneck_dir'),
                    options.get('image_dir'),
                    jpeg_data_tensor,
                    bottleneck_tensor
                )
            )
            test_accuracy, predictions = sess.run(
                [
                    evaluation_step,
                    prediction
                ],
                feed_dict={
                    bottleneck_input: test_bottlenecks,
                    ground_truth_input: test_ground_truth
                }
            )
            print('Final test accuracy = %.1f%% (N=%d)' % (
                test_accuracy * 100,
                len(test_bottlenecks)
            ))

            if options.get('print_misclassified_test_images'):
                print('=== MISCLASSIFIED TEST IMAGES ===')
                for i, test_filename in enumerate(test_filenames):
                    if predictions[i] != test_ground_truth[i].argmax():
                        print('%70s  %s' % (
                            test_filename,
                            list(image_lists.keys())[predictions[i]]
                        ))

            # Write out the trained graph and labels with the weights stored as
            # constants.
            output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                graph.as_graph_def(),
                [
                    options.get('final_tensor_name')
                ]
            )

            with gfile.FastGFile(options.get('output_graph'), 'wb') as f:
                f.write(output_graph_def.SerializeToString())

            with gfile.FastGFile(options.get('output_labels'), 'w') as f:
                f.write('\n'.join(image_lists.keys()) + '\n')
