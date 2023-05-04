python run_clm.py \
    --train_file ../ps-interview_/data/processed_clm/train.json \
    --validation_file ../ps-interview_/data/processed_clm/valid.json \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --model_name_or_path gpt2 \
    --output_dir ../ps-interview_/dump/gpt2_1.0