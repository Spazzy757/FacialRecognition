#!/usr/bin/env bash
# Todo: Change to django management command  settings move to global_settings
python engine/management/commands/retraining.py \
--image_dir=/code/media/dataset/ \
--model_dir=/code/media/models/ \
--output_labels=/code/media/models/breed_labels.txt \
--bottleneck_dir=media/bottleknecks \
--summaries_dir=/code/media/summaries/ \
--output_graph=/code/media/models/breed_model.pb \
-- eval_step_interval=10 \
--flip_left_right \
--random_scale=0 \
--random_brightness=10 \
--random_crop=10 \


# --output_labels
# --summaries_dir
# --how_many_training_steps
# --learning_rate
# --validation_percentage
# --train_batch_size
# --test_batch_size
# --validation_batch_size
# --print_misclassified_test_images
# --model_dir
# --bottleneck_dir
# --final_tensor_name

# These are used to increase the diversity (scale) the dataset
# --flip_left_right
# --random_scale
# --random_brightness
# --random_crop

# This is how often to print out
# --eval_step_interval