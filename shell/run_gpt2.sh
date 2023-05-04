python run_llama.py \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --model_name_or_path gpt2 \
    --output_dir ../ps_interview/dump/gpt2_1.0
