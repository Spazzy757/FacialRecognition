#!/usr/bin/env bash
# Todo: Change this into a django management command
python Tensorflow-Dog-Breed-Classifier/retraining.py \
--image_dir=/code/media/dataset/ \
--model_dir=/code/media/models/ \
--output_labels=/code/media/models/breed_labels.txt \
--bottleneck_dir=media/bottleknecks \
--output_graph=/code/media/models/breed_model.pb

# --output_labels
# --summaries_dir
# --how_many_training_steps
# --learning_rate
# --validation_percentage
# --eval_step_interval
# --train_batch_size
# --test_batch_size
# --validation_batch_size
# --print_misclassified_test_images
# --model_dir
# --bottleneck_dir
# --final_tensor_name
# --flip_left_right
# --random_crop
# --random_scale
# --random_brightness
